"""
File handling utilities
"""
from pathlib import Path

def search_for_pdf(filename: str) -> str:
    """
    Search for a PDF file in common locations.
    
    Args:
        filename: Name of the PDF file to search for
        
    Returns:
        str: Full path to the file if found, None otherwise
    """
    if not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"
    
    search_locations = [
        Path.cwd(),
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home() / "Downloads",
    ]
    
    for location in search_locations:
        if location.exists():
            potential_path = location / filename
            if potential_path.exists() and potential_path.is_file():
                return str(potential_path)
    
    return None