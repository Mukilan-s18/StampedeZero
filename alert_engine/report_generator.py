import os
import datetime
from fpdf import FPDF

def generate_incident_report(timestamp, peak_inflow, total_size, image_path):
    """
    Generates a 1-page PDF incident report for legal and insurance purposes.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Fonts
    pdf.set_font("Arial", 'B', 24)
    
    # Title
    pdf.set_text_color(220, 50, 50)
    pdf.cell(200, 20, txt="StampedeZero Automated Incident Report", ln=1, align='C')
    
    # Horizontal line
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Metadata
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(50, 10, txt="Incident Details:", ln=1)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Time of Alert: {timestamp}", ln=1)
    pdf.cell(200, 10, txt=f"Peak Inflow Rate: {peak_inflow:.1f} ppl/sec", ln=1)
    pdf.cell(200, 10, txt=f"Estimated Total Crowd Size: {total_size} people", ln=1)
    pdf.cell(200, 10, txt=f"Location: Venue Zone Alpha (Camera 1)", ln=1)
    
    pdf.ln(10)
    
    # Image Evidence
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Photographic Evidence (Heatmap Snapshot):", ln=1)
    
    if os.path.exists(image_path):
        pdf.image(image_path, x=15, w=180)
    else:
        pdf.set_font("Arial", 'I', 12)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(200, 10, txt="[Snapshot image not found]", ln=1)
        
    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, 'Generated automatically by StampedeZero AI Proactive Defense System.', 0, 0, 'C')
    
    # Save PDF
    output_filename = f"Incident_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_filename)
    
    return output_filename
