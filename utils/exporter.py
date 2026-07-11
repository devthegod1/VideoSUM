import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def export_to_txt(text_content: str) -> BytesIO:
    """Converts string content into a temporary plain text binary buffer for download."""
    buffer = BytesIO()
    buffer.write(text_content.encode("utf-8"))
    buffer.seek(0)
    return buffer

def export_to_pdf(markdown_content: str, document_title: str = "Meeting Minutes") -> BytesIO:
    """
    Parses structural markdown headings and text blocks into a beautiful corporate PDF layout.
    
    Returns:
        BytesIO: A binary buffer containing the generated PDF data.
    """
    buffer = BytesIO()
    
    # 1. Setup Document Layout Blueprint
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,  # Standard 0.75-inch margins
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 2. Establish Corporate Typography Hierarchy
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        spaceAfter=15,
        textColor='#1A365D' # Warm Corporate Navy
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
        textColor='#2B6CB0' # Accent Blue
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        spaceAfter=8,
        alignment=TA_LEFT
    )
    
    story = []
    
    # 3. Build Cover / Header
    story.append(Paragraph(document_title, title_style))
    story.append(HRFlowable(width="100%", thickness=2, color="#1A365D", spaceAfter=20))
    
    # 4. Parse Content Blocks Line-by-Line
    lines = markdown_content.split("\n")
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
            
        # Parse Headings (e.g., ## 📝 Executive Summary)
        if cleaned_line.startswith("##"):
            # Strip markdown notation symbols
            clean_text = cleaned_line.replace("##", "").replace("**", "").strip()
            story.append(Spacer(1, 10))
            story.append(Paragraph(clean_text, h2_style))
        
        # Parse standard content or bullet structures
        else:
            # Clean out common inline bold structures if returned by the LLM
            clean_text = cleaned_line.replace("**", "")
            story.append(Paragraph(clean_text, body_style))
            
    # 5. Compile Everything into the Binary Stream
    doc.build(story)
    buffer.seek(0)
    return buffer