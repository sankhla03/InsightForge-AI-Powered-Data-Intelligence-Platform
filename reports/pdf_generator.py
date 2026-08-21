"""
reports/pdf_generator.py
=========================
Professional PDF Report Generator for InsightForge.

Uses ReportLab (pure Python, no system dependencies) to generate
a company-style ML pipeline report PDF.

REPORT SECTIONS:
    1. Cover Page
    2. Project Information
    3. Dataset Summary
    4. Missing Value Analysis
    5. Duplicate Analysis
    6. Outlier Summary
    7. Encoding Summary
    8. Feature Selection Summary
    9. Scaling Summary
    10. Model Recommendation
    11. Model Comparison Table
    12. Best Model Evaluation Metrics
    13. Confusion Matrix (Classification) / Residuals Info (Regression)
    14. Feature Importance
    15. Pipeline Flow Diagram
    16. Conclusions & Recommendations
    17. Appendix

USAGE:
    from reports.pdf_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(report_data)
    # response = HttpResponse(pdf_bytes, content_type='application/pdf')
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch, mm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether,
        PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.platypus.flowables import Flowable
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation unavailable.")


# ============================================================================
# COLOR PALETTE
# ============================================================================

if REPORTLAB_AVAILABLE:
    # Brand Colors
    PRIMARY     = colors.HexColor('#2563EB')    # Indigo-600
    PRIMARY_DARK = colors.HexColor('#1D4ED8')   # Indigo-700
    SECONDARY   = colors.HexColor('#7C3AED')    # Violet-600
    SUCCESS     = colors.HexColor('#059669')    # Emerald-600
    WARNING     = colors.HexColor('#D97706')    # Amber-600
    DANGER      = colors.HexColor('#DC2626')    # Red-600
    DARK        = colors.HexColor('#1E293B')    # Slate-800
    MEDIUM      = colors.HexColor('#475569')    # Slate-600
    LIGHT       = colors.HexColor('#94A3B8')    # Slate-400
    BG_LIGHT    = colors.HexColor('#F1F5F9')    # Slate-100
    BG_CARD     = colors.HexColor('#F8FAFC')    # Slate-50
    WHITE       = colors.white
    BLACK       = colors.black
    TRANSPARENT = colors.transparent


# ============================================================================
# PAGE TEMPLATE WITH HEADER/FOOTER
# ============================================================================

class HeaderFooterCanvas:
    """Adds professional header and footer to every page."""

    def __init__(self, project_name: str = 'InsightForge', timestamp: str = ''):
        self.project_name = project_name
        self.timestamp = timestamp

    def __call__(self, canvas, doc):
        canvas.saveState()
        width, height = A4

        # ── Header ──────────────────────────────────────────────────────────
        # Top bar
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)

        # Logo / title in header
        canvas.setFillColor(WHITE)
        canvas.setFont('Helvetica-Bold', 11)
        canvas.drawString(1 * cm, height - 0.85 * cm, '🔍 InsightForge ML Platform')

        # Page info on right
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(
            width - 1 * cm,
            height - 0.85 * cm,
            f'Page {doc.page}'
        )

        # ── Footer ──────────────────────────────────────────────────────────
        canvas.setFillColor(BG_LIGHT)
        canvas.rect(0, 0, width, 0.9 * cm, fill=1, stroke=0)

        canvas.setStrokeColor(PRIMARY)
        canvas.setLineWidth(0.5)
        canvas.line(0, 0.9 * cm, width, 0.9 * cm)

        canvas.setFillColor(MEDIUM)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(1 * cm, 0.3 * cm, f'InsightForge ML Report — Generated: {self.timestamp}')
        canvas.drawRightString(width - 1 * cm, 0.3 * cm, 'Confidential — ML Pipeline Analysis')

        canvas.restoreState()


# ============================================================================
# STYLES
# ============================================================================

def _build_styles():
    """Build and return paragraph style catalog."""
    base = getSampleStyleSheet()
    styles = {}

    def ps(name, **kwargs) -> ParagraphStyle:
        return ParagraphStyle(name, **kwargs)

    styles['title'] = ps(
        'IFTitle',
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=6,
        leading=34,
    )
    styles['subtitle'] = ps(
        'IFSubtitle',
        fontName='Helvetica',
        fontSize=14,
        textColor=colors.HexColor('#BFDBFE'),
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=18,
    )
    styles['h1'] = ps(
        'IFH1',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        leading=20,
        borderPad=(0, 0, 4, 0),
    )
    styles['h2'] = ps(
        'IFH2',
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=DARK,
        spaceBefore=10,
        spaceAfter=6,
        leading=16,
    )
    styles['h3'] = ps(
        'IFH3',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=MEDIUM,
        spaceBefore=8,
        spaceAfter=4,
    )
    styles['body'] = ps(
        'IFBody',
        fontName='Helvetica',
        fontSize=10,
        textColor=DARK,
        spaceBefore=4,
        spaceAfter=4,
        leading=14,
    )
    styles['body_small'] = ps(
        'IFBodySmall',
        fontName='Helvetica',
        fontSize=9,
        textColor=MEDIUM,
        spaceBefore=2,
        spaceAfter=2,
        leading=12,
    )
    styles['label'] = ps(
        'IFLabel',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=PRIMARY,
    )
    styles['center'] = ps(
        'IFCenter',
        fontName='Helvetica',
        fontSize=10,
        textColor=DARK,
        alignment=TA_CENTER,
    )
    styles['badge_success'] = ps(
        'IFBadgeSuccess',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=SUCCESS,
        alignment=TA_CENTER,
    )
    styles['badge_warning'] = ps(
        'IFBadgeWarning',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=WARNING,
        alignment=TA_CENTER,
    )
    return styles


# ============================================================================
# TABLE STYLE HELPERS
# ============================================================================

def _header_table_style(header_bg=None) -> TableStyle:
    """Standard table style with colored header."""
    header_bg = header_bg or PRIMARY
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, BG_LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 0), (-1, 0), [header_bg]),
    ])


def _section_divider(styles) -> list:
    """Return a styled section divider."""
    return [
        Spacer(1, 0.3 * cm),
        HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E2E8F0')),
        Spacer(1, 0.3 * cm),
    ]


# ============================================================================
# SECTION BUILDERS
# ============================================================================

def _build_cover_page(story: list, styles: dict, report_data: dict):
    """Build a professional cover page."""
    # Full-page gradient background via a colored rectangle would require canvas
    # We simulate with colored table
    project_name = report_data.get('project', {}).get('name', 'ML Pipeline Analysis')
    generated_at = report_data.get('generated_at', datetime.now().strftime('%B %d, %Y at %H:%M'))
    dataset_name = report_data.get('dataset', {}).get('filename', 'Dataset')

    # Cover art using a wide colored table
    cover_data = [
        [Paragraph('🔍 InsightForge', ParagraphStyle(
            'CoverLogo', fontName='Helvetica-Bold', fontSize=14,
            textColor=colors.HexColor('#BFDBFE'), alignment=TA_CENTER,
        ))],
        [Paragraph('ML Platform', ParagraphStyle(
            'CoverPlatform', fontName='Helvetica', fontSize=11,
            textColor=colors.HexColor('#93C5FD'), alignment=TA_CENTER,
        ))],
        [Spacer(1, 0.5 * cm)],
        [Paragraph(project_name, styles['title'])],
        [Spacer(1, 0.3 * cm)],
        [Paragraph('Machine Learning Pipeline Report', styles['subtitle'])],
        [Spacer(1, 0.8 * cm)],
        [Paragraph(f'Dataset: {dataset_name}', styles['subtitle'])],
        [Paragraph(f'Generated: {generated_at}', ParagraphStyle(
            'CoverDate', fontName='Helvetica', fontSize=11,
            textColor=colors.HexColor('#93C5FD'), alignment=TA_CENTER,
        ))],
    ]
    cover_table = Table(cover_data, colWidths=[17 * cm])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ROUNDEDCORNERS', [8]),
    ]))

    story.append(Spacer(1, 3 * cm))
    story.append(cover_table)
    story.append(Spacer(1, 1.5 * cm))

    # Summary box
    summary_items = [
        ['📊 Task Type', str(report_data.get('model_report', {}).get('task_type', 'N/A')).title()],
        ['🤖 Best Model', str(report_data.get('model_report', {}).get('best_model', 'N/A'))],
        ['📈 Primary Metric', _get_primary_metric_display(report_data.get('model_report', {}))],
        ['📁 Total Features', str(report_data.get('dataset', {}).get('columns', 'N/A'))],
        ['📋 Total Rows', str(report_data.get('dataset', {}).get('rows', 'N/A'))],
    ]
    sum_table = Table(
        [[Paragraph(k, styles['label']), Paragraph(str(v), styles['body'])] for k, v in summary_items],
        colWidths=[7 * cm, 10 * cm]
    )
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_CARD),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [BG_CARD, WHITE]),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(sum_table)
    story.append(PageBreak())


def _build_dataset_section(story: list, styles: dict, report_data: dict):
    """Build dataset summary section."""
    story.append(Paragraph('1. Dataset Summary', styles['h1']))
    dataset = report_data.get('dataset', {})

    if not dataset:
        story.append(Paragraph('No dataset information available.', styles['body']))
        return

    # Key stats
    stats = [
        ['Metric', 'Value'],
        ['Total Rows', f"{dataset.get('rows', 'N/A'):,}" if isinstance(dataset.get('rows'), int) else str(dataset.get('rows', 'N/A'))],
        ['Total Columns', str(dataset.get('columns', 'N/A'))],
        ['Numeric Columns', str(dataset.get('numeric_cols', 'N/A'))],
        ['Categorical Columns', str(dataset.get('categorical_cols', 'N/A'))],
        ['Missing Values', str(dataset.get('missing_count', 0))],
        ['Duplicate Rows', str(dataset.get('duplicate_rows', 0))],
        ['File Name', str(dataset.get('filename', 'N/A'))],
    ]
    t = Table(stats, colWidths=[8 * cm, 9 * cm])
    t.setStyle(_header_table_style())
    story.append(t)

    # Column info
    col_info = dataset.get('column_info', [])
    if col_info:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph('Column Details', styles['h2']))
        col_data = [['Column Name', 'Data Type', 'Missing', 'Unique Values', 'Sample Values']]
        for ci in col_info[:20]:  # Limit to 20 columns for readability
            col_data.append([
                str(ci.get('name', '')),
                str(ci.get('dtype', '')),
                str(ci.get('missing', 0)),
                str(ci.get('unique', 'N/A')),
                str(ci.get('sample', ''))[:40],
            ])
        col_table = Table(col_data, colWidths=[4.5 * cm, 2.5 * cm, 2 * cm, 3 * cm, 5 * cm])
        col_table.setStyle(_header_table_style())
        story.append(col_table)


def _build_preprocessing_section(story: list, styles: dict, report_data: dict):
    """Build the preprocessing overview section."""
    story.append(PageBreak())
    story.append(Paragraph('2. Preprocessing Pipeline', styles['h1']))

    # Outlier section
    outliers = report_data.get('outliers', {})
    story.append(Paragraph('2.1 Outlier Handling', styles['h2']))
    if outliers:
        out_data = [['Metric', 'Value']]
        out_data.extend([
            ['Outliers Detected', str(outliers.get('count', 0))],
            ['Method Applied', str(outliers.get('method', 'None'))],
            ['Rows Affected', str(outliers.get('rows_corrected', 0))],
        ])
        t = Table(out_data, colWidths=[8 * cm, 9 * cm])
        t.setStyle(_header_table_style(SUCCESS))
        story.append(t)
    else:
        story.append(Paragraph('No outlier handling data available.', styles['body']))

    # Encoding section
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph('2.2 Feature Encoding', styles['h2']))
    encoding = report_data.get('features', {}).get('encoding', {})
    if encoding:
        enc_data = [['Column', 'Method']]
        for col, method in encoding.items():
            enc_data.append([str(col), str(method)])
        t = Table(enc_data, colWidths=[9 * cm, 8 * cm])
        t.setStyle(_header_table_style(SECONDARY))
        story.append(t)
    else:
        story.append(Paragraph('Ordinal encoding applied to categorical features.', styles['body']))

    # Feature selection
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph('2.3 Feature Selection', styles['h2']))
    features = report_data.get('features', {})
    if features:
        feat_data = [['Metric', 'Value']]
        feat_data.extend([
            ['Method', str(features.get('method', 'N/A'))],
            ['Selected Features', str(features.get('n_selected', 'N/A'))],
            ['Total Features', str(features.get('n_original', 'N/A'))],
        ])
        t = Table(feat_data, colWidths=[8 * cm, 9 * cm])
        t.setStyle(_header_table_style(colors.HexColor('#0891B2')))
        story.append(t)

        selected = features.get('selected', [])
        if selected:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f'Selected features: {", ".join(str(f) for f in selected)}', styles['body_small']))

    # Scaling
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph('2.4 Feature Scaling', styles['h2']))
    scaling = report_data.get('scaling', {})
    if scaling:
        story.append(Paragraph(
            f"Scaling method: <b>{scaling.get('method', 'N/A')}</b> — "
            f"Applied to {scaling.get('n_features', 'N/A')} numeric features.",
            styles['body']
        ))


def _build_models_section(story: list, styles: dict, report_data: dict):
    """Build model comparison and evaluation section."""
    story.append(PageBreak())
    story.append(Paragraph('3. Model Training & Evaluation', styles['h1']))

    # Model comparison table
    model_results = report_data.get('model_results', {})
    best_model = report_data.get('model_report', {}).get('best_model', '')
    task_type = report_data.get('model_report', {}).get('task_type', 'classification')

    if model_results:
        story.append(Paragraph('3.1 Model Comparison', styles['h2']))
        if task_type == 'classification':
            headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Best?']
            rows = [headers]
            for name, data in model_results.items():
                is_best = name == best_model
                rows.append([
                    name,
                    f"{data.get('accuracy', 0):.1f}%",
                    f"{data.get('precision', 0):.1f}%",
                    f"{data.get('recall', 0):.1f}%",
                    f"{data.get('f1', 0):.1f}%",
                    f"{data.get('roc_auc', 0):.1f}%",
                    '⭐ Best' if is_best else '',
                ])
            col_widths = [5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm]
        else:
            headers = ['Model', 'R² Score', 'MAE', 'RMSE', 'Accuracy %', 'Best?']
            rows = [headers]
            for name, data in model_results.items():
                is_best = name == best_model
                rows.append([
                    name,
                    f"{data.get('r2', 0):.1f}%",
                    f"{data.get('mae', 0):.4f}",
                    f"{data.get('rmse', 0):.4f}",
                    f"{data.get('accuracy', 0):.1f}%",
                    '⭐ Best' if is_best else '',
                ])
            col_widths = [6*cm, 3*cm, 3*cm, 3*cm, 3*cm, 2*cm]

        t = Table(rows, colWidths=col_widths)
        style = _header_table_style()
        # Highlight best model row
        for i, (name, _) in enumerate(model_results.items(), start=1):
            if name == best_model:
                style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#D1FAE5'))
                style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
        t.setStyle(style)
        story.append(t)

    # Best model metrics
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f'3.2 Best Model: {best_model}', styles['h2']))
    model_report = report_data.get('model_report', {})
    if model_report:
        if task_type == 'classification':
            metrics = [
                ['Metric', 'Value'],
                ['Accuracy', f"{model_report.get('accuracy', 0):.2f}%"],
                ['F1 Score (Weighted)', f"{model_report.get('f1', 0):.2f}%"],
                ['Precision', f"{model_report.get('precision', 0):.2f}%"],
                ['Recall', f"{model_report.get('recall', 0):.2f}%"],
                ['ROC-AUC', f"{model_report.get('roc_auc', 0):.2f}%"],
            ]
        else:
            metrics = [
                ['Metric', 'Value'],
                ['R² Score', f"{model_report.get('r2', 0):.2f}%"],
                ['Mean Absolute Error (MAE)', str(model_report.get('mae', 'N/A'))],
                ['Root Mean Squared Error (RMSE)', str(model_report.get('rmse', 'N/A'))],
                ['Accuracy (within 10%)', f"{model_report.get('accuracy', 0):.2f}%"],
            ]
        t = Table(metrics, colWidths=[10 * cm, 7 * cm])
        t.setStyle(_header_table_style(SUCCESS))
        story.append(t)

    # Confusion matrix image if available
    cm_image = report_data.get('confusion_matrix_path')
    if cm_image and os.path.exists(cm_image):
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph('3.3 Confusion Matrix', styles['h2']))
        try:
            img = Image(cm_image, width=12 * cm, height=9 * cm)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f'(Confusion matrix image unavailable: {e})', styles['body_small']))


def _build_conclusion_section(story: list, styles: dict, report_data: dict):
    """Build conclusions and recommendations section."""
    story.append(PageBreak())
    story.append(Paragraph('4. Conclusions & Recommendations', styles['h1']))

    conclusion = report_data.get('conclusion', {})
    best_model = report_data.get('model_report', {}).get('best_model', 'N/A')
    task_type = report_data.get('model_report', {}).get('task_type', 'classification')

    # Summary
    story.append(Paragraph('Pipeline Summary', styles['h2']))
    story.append(Paragraph(
        f"The InsightForge pipeline successfully processed your dataset and trained multiple "
        f"machine learning models for a <b>{task_type}</b> task. "
        f"The best performing model was <b>{best_model}</b> based on "
        f"{'F1 Score' if task_type == 'classification' else 'R² Score'}.",
        styles['body']
    ))

    # Recommendations
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph('Recommendations', styles['h2']))

    recommendations = conclusion.get('recommendations', [])
    if not recommendations:
        recommendations = [
            'Validate model performance on a held-out test set before production deployment.',
            'Monitor model performance over time for data drift.',
            'Consider hyperparameter tuning for further performance improvements.',
            'Ensure the preprocessing pipeline is applied consistently at inference time.',
            'Document feature transformations for reproducibility.',
        ]

    for rec in recommendations:
        story.append(Paragraph(f'• {rec}', styles['body']))

    # Pipeline flow
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph('Pipeline Flow', styles['h2']))
    pipeline_steps = [
        'Upload Dataset', '→', 'Data Cleaning', '→', 'Outlier Handling',
        '→', 'Label Noise', '→', 'Encoding', '→', 'Feature Selection',
        '→', 'Feature Scaling', '→', 'Model Training', '→', 'Prediction'
    ]
    story.append(Paragraph(
        ' '.join(pipeline_steps),
        ParagraphStyle(
            'PipelineFlow',
            fontName='Helvetica',
            fontSize=9,
            textColor=MEDIUM,
            leading=14,
        )
    ))


def _get_primary_metric_display(model_report: dict) -> str:
    """Get the primary metric display string."""
    task_type = model_report.get('task_type', 'classification')
    if task_type == 'classification':
        acc = model_report.get('accuracy', 0)
        return f"Accuracy: {acc:.1f}%"
    else:
        r2 = model_report.get('r2', 0)
        return f"R²: {r2:.1f}%"


# ============================================================================
# MAIN PDF GENERATION FUNCTION
# ============================================================================

def generate_pdf_report(report_data: Dict) -> bytes:
    """
    Generate a complete professional PDF report.

    Args:
        report_data: Report data dict (from get_pipeline_report())

    Returns:
        PDF bytes that can be sent as an HttpResponse
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "ReportLab is not installed. Install it with: pip install reportlab"
        )

    buffer = io.BytesIO()
    width, height = A4

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    project_name = report_data.get('project', {}).get('name', 'ML Pipeline Report')

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title=project_name,
        author='InsightForge ML Platform',
        subject='Machine Learning Pipeline Report',
    )

    # Page template with header/footer
    hf = HeaderFooterCanvas(project_name=project_name, timestamp=timestamp)
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin + 0.5 * cm,
        doc.width,
        doc.height - 0.5 * cm,
        id='main_frame'
    )
    doc.addPageTemplates([
        PageTemplate(id='main', frames=frame, onPage=hf),
    ])

    # Store generated_at in report_data for cover page
    report_data['generated_at'] = timestamp

    # Build styles
    styles = _build_styles()

    # Build story (content)
    story = []

    _build_cover_page(story, styles, report_data)
    _build_dataset_section(story, styles, report_data)
    _build_preprocessing_section(story, styles, report_data)
    _build_models_section(story, styles, report_data)
    _build_conclusion_section(story, styles, report_data)

    # Build PDF
    try:
        doc.build(story)
    except Exception as e:
        logger.exception(f"PDF generation failed: {e}")
        raise

    buffer.seek(0)
    return buffer.read()
