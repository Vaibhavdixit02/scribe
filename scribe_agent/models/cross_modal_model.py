import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import einops
from transformers import BitsAndBytesConfig

class ElementLocalizationModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.localization_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, hidden_states, original_states=None):
        """
        Compute localization scores for each token
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            original_states: Original hidden states before cross-attention
            
        Returns:
            Tensor of shape (batch_size, seq_len) with scores for each token
        """
        # Add residual connection if original states provided
        if original_states is not None:
            hidden_states = hidden_states + original_states
            
        # Compute localization scores
        scores = self.localization_head(hidden_states).squeeze(-1)
        
        return scores

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Cross-attention layers
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, text_features, visual_features, attention_mask=None):
        """
        Apply cross-attention from text to visual features
        
        Args:
            text_features: Tensor of shape (batch_size, text_seq_len, hidden_size)
            visual_features: Tensor of shape (batch_size, visual_seq_len, hidden_size)
            attention_mask: Optional mask for text features
            
        Returns:
            Tensor of shape (batch_size, text_seq_len, hidden_size)
        """
        batch_size, text_seq_len, _ = text_features.shape
        _, visual_seq_len, _ = visual_features.shape
        
        # Project queries, keys, values
        q = self.query(text_features)
        k = self.key(visual_features)
        v = self.value(visual_features)
        
        # Reshape for multi-head attention
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for batch size and heads
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = einops.rearrange(context, 'b h s d -> b s (h d)')
        
        # Apply output projection
        output = self.output(context)
        
        return output

class CrossModalWebAgent(nn.Module):
    def __init__(self, 
                 text_model_name="Qwen/Qwen2-7B", 
                 vision_model_name="openai/clip-vit-base-patch32",
                 use_lora=True,
                 lora_rank=16,
                 lora_alpha=32,
                ):
        super().__init__()
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base text model (Qwen2-7B)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name, 
            torch_dtype=torch.float16, 
            # device_map="auto", 
            quantization_config=quantization_config
        )
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(
            vision_model_name, 
            torch_dtype=torch.float16, 
            # device_map="auto",
            quantization_config=quantization_config
        )
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        
        # Get hidden sizes
        self.text_hidden_size = self.text_model.config.hidden_size
        self.vision_hidden_size = self.clip_model.config.vision_config.hidden_size
        
        # Projection layer for visual features
        self.visual_projection = nn.Linear(
            self.vision_hidden_size, 
            self.text_hidden_size
        )
        
        # Cross-attention layer
        self.cross_attention = CrossModalAttention(
            hidden_size=self.text_hidden_size,
            num_heads=8
        )
        
        # Element localization module
        self.element_localization = ElementLocalizationModule(
            hidden_size=self.text_hidden_size
        )
        
        # Apply LoRA if requested
        if use_lora:
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.text_model = get_peft_model(self.text_model, peft_config)
    
    def preprocess_image(self, images):
        """
        Preprocess images using the CLIP processor
        
        Args:
            images: List of image data (can be PIL, numpy arrays, tensors)
            
        Returns:
            Preprocessed image tensors
        """
        inputs = self.clip_processor(images=images, return_tensors="pt")
        return inputs
    
    def encode_images(self, images=None, pixel_values=None):
        """
        Encode images using the CLIP model.
        
        Args:
            images: Raw image data (will be preprocessed if provided)
            pixel_values: Already preprocessed pixel values
            
        Returns:
            Encoded image features projected to the text model's dimension
        """
        # Preprocess images if raw images provided
        if images is not None and pixel_values is None:
            inputs = self.preprocess_image(images)
            pixel_values = inputs["pixel_values"].to(next(self.clip_model.parameters()).device)
        
        # Use provided pixel_values if images not provided
        if pixel_values is not None:
            # Extract image features using CLIP's get_image_features method
            image_features = self.clip_model.get_image_features(pixel_values)
            
            # Convert the pooled features to sequence format for cross-attention
            # We need to reshape the features as a sequence of 1 token with the embedding dimension
            batch_size = image_features.shape[0]
            image_features = image_features.view(batch_size, 1, -1)
            
            # Project to text model's dimension
            projected_features = self.visual_projection(image_features)
            
            return projected_features
        
        return None
    
    def process_visual_features(self, visual_features_dict):
        """
        Process the visual features dictionary containing multiple images.
        
        Args:
            visual_features_dict: Dictionary with 'full_image' and 'elements' entries
            
        Returns:
            Processed visual features tensor
        """
        # Process full image
        if isinstance(visual_features_dict['full_image'], dict) and 'pixel_values' in visual_features_dict['full_image']:
            full_image_pixel_values = visual_features_dict['full_image']['pixel_values']
        else:
            full_image_pixel_values = visual_features_dict['full_image'][0]['pixel_values']
            
        full_image_features = self.encode_images(pixel_values=full_image_pixel_values)
        
        # Process individual elements (if available)
        element_features = []
        if 'elements' in visual_features_dict and visual_features_dict['elements']:
            for element_id, element_inputs in visual_features_dict['elements'].items():
                if isinstance(element_inputs, dict) and 'pixel_values' in element_inputs:
                    element_pixel_values = element_inputs['pixel_values']
                else:
                    element_pixel_values = element_inputs[0]['pixel_values']
                
                element_feature = self.encode_images(pixel_values=element_pixel_values)
                element_features.append(element_feature)
            
            if element_features:
                # Concatenate all element features
                element_features = torch.cat(element_features, dim=1)
                # Concatenate with full image features
                all_features = torch.cat([full_image_features, element_features], dim=1)
            else:
                all_features = full_image_features
        else:
            all_features = full_image_features
            
        return all_features
    
    def forward(self, input_ids, attention_mask, images=None, visual_features=None, labels=None):
        """
        Forward pass for the cross-modal web agent
        
        Args:
            input_ids: Token IDs for text input
            attention_mask: Attention mask for text input
            images: Raw images (will be preprocessed and encoded)
            visual_features: Pre-processed visual features
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary with model outputs
        """
        # Process text input
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        if hasattr(text_outputs, 'hidden_states'):
            text_hidden_states = text_outputs.hidden_states[-1]
        else:
            # For models where hidden states are not returned directly
            # This is a fallback that may need to be adjusted based on the model
            text_hidden_states = self.text_model.transformer.output_hidden_states(
                text_outputs.logits, 
                input_ids
            )
        
        # Process visual input if provided
        processed_visual_features = None
        
        if images is not None:
            # Encode raw images
            processed_visual_features = self.encode_images(images=images)
        elif visual_features is not None:
            # Process pre-processed visual features
            if isinstance(visual_features, dict):
                processed_visual_features = self.process_visual_features(visual_features)
            else:
                processed_visual_features = visual_features
        
        # Apply cross-modal processing if visual features available
        if processed_visual_features is not None:
            # Apply cross-attention
            enhanced_text_features = self.cross_attention(
                text_features=text_hidden_states,
                visual_features=processed_visual_features,
                attention_mask=attention_mask
            )
            
            # Localize elements
            element_scores = self.element_localization(
                enhanced_text_features, 
                text_hidden_states
            )
            
            # Feed enhanced features back to language model
            lm_logits = self.text_model.lm_head(enhanced_text_features)
            
            result = {
                'logits': lm_logits,
                'element_scores': element_scores
            }
            
        else:
            # Text-only forward pass
            lm_logits = text_outputs.logits
            result = {'logits': lm_logits}
            
        # Calculate loss if labels provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            result['loss'] = loss
            
        return result
