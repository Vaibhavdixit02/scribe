import re
from bs4 import BeautifulSoup, Comment

def collect_tags(tag, all_tags):
    """Recursively collect all tags in a BeautifulSoup object."""
    all_tags.append(tag)
    for child in tag.find_all(recursive=False):
        collect_tags(child, all_tags)

def remove_comments(soup):
    """Remove HTML comments from BeautifulSoup object."""
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

def remove_scripts_and_styles(soup):
    """Remove script and style tags from BeautifulSoup object."""
    for tag in soup(['script', 'style', 'noscript', 'svg', 'path']):
        tag.decompose()

def prune_attributes(soup, keep_attrs=None):
    """Remove non-essential attributes from all tags."""
    if keep_attrs is None:
        keep_attrs = ['node', 'backend_node_id', 'href', 'src', 'value', 'placeholder', 
                      'aria-label', 'title', 'alt', 'type', 'name', 'id', 'class', 'role']
    
    for tag in soup.find_all():
        attrs_to_remove = [attr for attr in tag.attrs if attr not in keep_attrs]
        for attr in attrs_to_remove:
            del tag[attr]

def truncate_dom(soup, max_tokens=28000):
    """Truncate DOM to fit within context window."""
    # Simple heuristic: estimate tokens by characters and prune less important branches
    content = str(soup)
    if len(content) <= max_tokens * 4:  # Rough estimate of 4 chars per token
        return content
    
    # Remove hidden elements first
    for tag in soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden')):
        tag.decompose()
    
    # If still too large, use tag-level truncation strategies
    tags_to_truncate = ['div', 'section', 'footer', 'aside', 'nav']
    for tag_name in tags_to_truncate:
        if len(str(soup)) <= max_tokens * 4:
            break
        
        # Keep removing div sections from the bottom until we're under the limit
        tags = soup.find_all(tag_name)
        for tag in reversed(tags):
            tag.decompose()
            if len(str(soup)) <= max_tokens * 4:
                break
    
    return str(soup)

def process_html(html_content, target_ids_backend=None):
    """Process HTML content to prepare it for the web agent."""
    # Create BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Collect all tags
    all_tags = []
    collect_tags(soup, all_tags)
    
    # Assign unique IDs
    for i, tag in enumerate(all_tags):
        tag["node"] = str(i)
        
    # Map backend IDs to node IDs
    full_map = {}
    for tag in all_tags:
        if "backend_node_id" in tag.attrs:
            full_map[tag["node"]] = tag["backend_node_id"]
            
    # Clean and prune DOM
    remove_comments(soup)
    remove_scripts_and_styles(soup)
    prune_attributes(soup)
    
    # Truncate to fit context window
    truncated_html = truncate_dom(soup)
    
    return truncated_html, full_map