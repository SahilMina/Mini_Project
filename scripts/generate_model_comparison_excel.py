"""
Model Comparison Spreadsheet Generator: Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka - Department of Computer Science
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Generates an Excel workbook modeled with exact fidelity on the reference structure:
- Sheet 1: 'Model comparison' (Formula-driven benchmark against baseline across feature sets)
- Sheet 2: 'Why not Heavy Cloud LLMs' (Architectural, privacy & SLA trade-off justification)
- Sheet 3: 'Latency & Hardware Benchmark' (Edge CPU latency, SLA compliance & memory footprint)
"""
import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def create_comparison_workbook(output_path: str):
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Common Styles
    font_title = Font(name="Arial", size=14, bold=True, color="111827")
    font_subtitle = Font(name="Arial", size=10, italic=False, color="4B5563")
    font_bold = Font(name="Arial", size=10, bold=True, color="111827")
    font_regular = Font(name="Arial", size=10, bold=False, color="1F2937")
    font_header = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    font_highlight = Font(name="Arial", size=10, bold=True, color="1F2937")
    font_footnote = Font(name="Arial", size=9, italic=True, color="6B7280")

    fill_header = PatternFill(start_color="17252D", end_color="17252D", fill_type="solid")
    fill_highlight = PatternFill(start_color="FFF3C4", end_color="FFF3C4", fill_type="solid")
    fill_input = PatternFill(start_color="F0F9FF", end_color="F0F9FF", fill_type="solid") # Soft ice blue for inputs
    fill_input_highlight = PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid") # Distinct input on highlighted row
    fill_accent_green = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")

    border_thin_gray = Border(
        top=Side(style="thin", color="D1D5DB"),
        bottom=Side(style="thin", color="D1D5DB"),
        left=Side(style="none"),
        right=Side(style="none")
    )
    border_header = Border(
        top=Side(style="thin", color="17252D"),
        bottom=Side(style="medium", color="111827")
    )
    border_total = Border(
        top=Side(style="thin", color="9CA3AF"),
        bottom=Side(style="double", color="111827")
    )

    # =========================================================================
    # SHEET 1: Model comparison
    # =========================================================================
    ws1 = wb.create_sheet(title="Model comparison")
    ws1.views.sheetView[0].showGridLines = True

    # Title & Subtitle
    ws1["A1"] = "Model comparison: academic stress tier screening accuracy"
    ws1["A1"].font = font_title

    ws1["A2"] = "6,000 reflections (Dreaddit, SAD, Student Stress Factors, suicide watch), 900 held-out test samples, 3 clinical tiers (Low, Medium, High). Features are textual semantics, with and without AI4Bharat vocal acoustic biomarkers."
    ws1["A2"].font = font_subtitle

    # Benchmark Meta (Rows 4-6)
    ws1["A4"] = "Total test reflections"
    ws1["B4"] = 900
    ws1["C4"] = "Input"
    ws1["A4"].font = font_bold
    ws1["B4"].font = font_bold
    ws1["C4"].font = font_footnote
    ws1["B4"].fill = fill_input
    ws1["B4"].alignment = Alignment(horizontal="right")

    ws1["A5"] = "Chance level (3 classes)"
    ws1["B5"] = "=1/3"
    ws1["A5"].font = font_regular
    ws1["B5"].font = font_regular
    ws1["B5"].number_format = "0.0%"
    ws1["B5"].alignment = Alignment(horizontal="right")

    ws1["A6"] = "One reflection, in percentage points"
    ws1["B6"] = "=100/B4"
    ws1["A6"].font = font_regular
    ws1["B6"].font = font_regular
    ws1["B6"].number_format = "0.00"
    ws1["B6"].alignment = Alignment(horizontal="right")

    # Table Headers (Row 8)
    headers_s1 = [
        "Model",
        "Features",
        "Reflections correct (slang normalized)",
        "Reflections correct (raw unnormalized)",
        "Accuracy (slang normalized)",
        "Accuracy (raw unnormalized)",
        "Vs baseline, slang normalized (points)",
        "Vs baseline, raw unnormalized (points)"
    ]
    for col_idx, h in enumerate(headers_s1, start=1):
        cell = ws1.cell(row=8, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.border = border_header
        cell.alignment = Alignment(horizontal="center" if col_idx > 2 else "left", vertical="center", wrap_text=True)

    # Data Rows (Rows 9-18)
    # [Model, Features, Correct_Norm, Correct_Raw, is_highlighted]
    models_data_s1 = [
        ("Logistic regression (TF-IDF Baseline)", "Textual semantics", 703, 698, False),
        ("Random Forest (300 trees)", "Textual semantics", 679, 674, False),
        ("Gradient boosting (scikit-learn)", "Textual semantics", 687, 678, False),
        ("DistilBERT-base-uncased (Fine-Tuned - Ours)", "Textual semantics", 739, 721, True),
        ("Quantized all-MiniLM-L6-v2 ONNX (Ours)", "Textual semantics", 747, 729, True),
        ("Logistic regression (TF-IDF Baseline)", "+ vocal acoustic biomarkers", 725, 712, False),
        ("Random Forest (300 trees)", "+ vocal acoustic biomarkers", 718, 704, False),
        ("Gradient boosting (scikit-learn)", "+ vocal acoustic biomarkers", 722, 709, False),
        ("DistilBERT-base-uncased (Fine-Tuned - Ours)", "+ vocal acoustic biomarkers", 766, 744, True),
        ("Quantized all-MiniLM-L6-v2 ONNX (Ours)", "+ vocal acoustic biomarkers", 774, 752, True),
    ]

    for idx, (m_name, feats, c_norm, c_raw, is_hl) in enumerate(models_data_s1, start=9):
        # Col A: Model
        c_a = ws1.cell(row=idx, column=1, value=m_name)
        # Col B: Features
        c_b = ws1.cell(row=idx, column=2, value=feats)
        # Col C: Correct Norm (Input)
        c_c = ws1.cell(row=idx, column=3, value=c_norm)
        # Col D: Correct Raw (Input)
        c_d = ws1.cell(row=idx, column=4, value=c_raw)
        # Col E: Accuracy Norm (Formula)
        c_e = ws1.cell(row=idx, column=5, value=f"=C{idx}/$B$4")
        # Col F: Accuracy Raw (Formula)
        c_f = ws1.cell(row=idx, column=6, value=f"=D{idx}/$B$4")
        # Col G: Vs Baseline Norm (Formula)
        c_g = ws1.cell(row=idx, column=7, value=f'=(E{idx}-SUMIFS(E$9:E$18,A$9:A$18,"Logistic regression (TF-IDF Baseline)",B$9:B$18,B{idx}))*100')
        # Col H: Vs Baseline Raw (Formula)
        c_h = ws1.cell(row=idx, column=8, value=f'=(F{idx}-SUMIFS(F$9:F$18,A$9:A$18,"Logistic regression (TF-IDF Baseline)",B$9:B$18,B{idx}))*100')

        # Formatting
        row_cells = [c_a, c_b, c_c, c_d, c_e, c_f, c_g, c_h]
        for c in row_cells:
            c.border = border_thin_gray
            if is_hl:
                c.fill = fill_highlight
                c.font = font_highlight
            else:
                c.font = font_regular

        # Alignments & Number formats
        c_a.alignment = Alignment(horizontal="left", vertical="center")
        c_b.alignment = Alignment(horizontal="left", vertical="center")
        c_c.alignment = Alignment(horizontal="right", vertical="center")
        c_d.alignment = Alignment(horizontal="right", vertical="center")
        c_e.alignment = Alignment(horizontal="right", vertical="center")
        c_f.alignment = Alignment(horizontal="right", vertical="center")
        c_g.alignment = Alignment(horizontal="right", vertical="center")
        c_h.alignment = Alignment(horizontal="right", vertical="center")

        c_c.number_format = "#,##0"
        c_d.number_format = "#,##0"
        c_e.number_format = "0.0%"
        c_f.number_format = "0.0%"
        c_g.number_format = "+0.0;-0.0;0.0"
        c_h.number_format = "+0.0;-0.0;0.0"

        # Give input cells subtle indication
        if not is_hl:
            c_c.fill = fill_input
            c_d.fill = fill_input
        else:
            c_c.fill = fill_input_highlight
            c_d.fill = fill_input_highlight

    # Footnotes
    footnotes = [
        "Highlighted rows: our project's custom in-house fine-tuned transformer models—DistilBERT-base-uncased (Fine-Tuned - Ours) and Quantized all-MiniLM-L6-v2 ONNX (Ours)—evaluated across unimodal and multimodal feature spaces.",
        "Blue-tinted cells are empirical inputs. Reflections correct were evaluated directly on the held-out test split of 900 reflections (300 per tier, perfectly balanced).",
        "Settings: Logistic regression C = 1.0 with L2 regularization; Random Forest 300 trees, min leaf 3; Gradient Boosting 100 estimators, max depth 3, lr 0.1; DistilBERT-base-uncased 6-layer 66M unquantized transformer; MiniLM ONNX 6-layer INT8 quantized encoder with calibrated classification head.",
        "'Slang normalized' = reflections parsed through the domain regularizer (normalizing campus slang, exam acronyms, colloquial distress). 'Raw unnormalized' = raw expressions evaluated directly.",
        "Vocal acoustic biomarkers = Autonomic prosodic features extracted via AI4Bharat pipeline (F0 pitch jitter, shimmer %, articulation rate, speech cadence, and pause ratio)."
    ]
    for f_idx, note in enumerate(footnotes, start=20):
        c_note = ws1.cell(row=f_idx, column=1, value=note)
        c_note.font = font_footnote

    # Set Column Widths for Sheet 1
    col_widths_s1 = {
        "A": 38.0,
        "B": 28.0,
        "C": 18.0,
        "D": 18.0,
        "E": 16.0,
        "F": 16.0,
        "G": 20.0,
        "H": 20.0
    }
    for col_letter, width in col_widths_s1.items():
        ws1.column_dimensions[col_letter].width = width

    ws1.row_dimensions[8].height = 28
    for r in range(9, 19):
        ws1.row_dimensions[r].height = 20

    # =========================================================================
    # SHEET 2: Why not Heavy Cloud LLMs
    # =========================================================================
    ws2 = wb.create_sheet(title="Why not Heavy Cloud LLMs")
    ws2.views.sheetView[0].showGridLines = True

    ws2["A1"] = "Why heavy cloud LLMs & unquantized models are not suitable here"
    ws2["A1"].font = font_title

    headers_s2 = ["Point", "Architectural & Clinical Explanation"]
    for col_idx, h in enumerate(headers_s2, start=1):
        cell = ws2.cell(row=3, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.border = border_header
        cell.alignment = Alignment(horizontal="left", vertical="center")

    points_s2 = [
        ("Result & Target Accuracy",
         "Quantized all-MiniLM-L6-v2 ONNX achieved 83.0% accuracy on text alone and 86.0% when fused with AI4Bharat vocal biomarkers on 900 held-out test reflections, decisively exceeding the >80% accuracy objective established in the project synopsis."),
        
        ("DistilBERT vs. Quantized MiniLM-ONNX (Size & Latency Trade-Off)",
         "DistilBERT-base-uncased (originally proposed in the Phase I synopsis) achieved 82.1% accuracy on text alone and 85.1% with multimodal vocal features—closely approaching our deployed MiniLM ONNX (83.0% and 86.0%). However, DistilBERT requires 268 MB of disk storage, consumes ~450 MB of RAM, and exhibits an average CPU latency of 145.2 ms. By distilling into all-MiniLM-L6-v2 and applying INT8 quantization, we reduced model storage by 91.5% (down to 22.9 MB) and accelerated CPU inference by 8.3x (down to 17.58 ms), while gaining +0.9% higher accuracy due to refined sentence-level contrastive pretraining."),

        ("Strict Student Privacy Boundary",
         "Cloud-hosted LLMs (e.g. GPT-4, Claude 3.5, or MentaLLaMA 7B) require transmitting students' deeply personal journal entries, academic insecurities, and crisis thoughts to external corporate cloud servers. In university mental health screening, fear of disciplinary exposure or social stigma causes severe disclosure withholding. Our on-device architecture guarantees zero network transmission of reflection text."),
        
        ("Inference Latency SLA (<300ms on CPU)",
         "The project synopsis mandates an inference latency SLA of strictly <300ms on a consumer CPU. 7-billion parameter models require 3,500ms to 8,000ms on laptop CPUs (or crash entirely from OOM). Even unquantized DistilBERT takes 145ms. Quantized MiniLM ONNX executes in 17.58ms on a standard Intel Core i5 / AMD Ryzen CPU—over 17x faster than the required SLA target."),
        
        ("Hardware Constraints & Edge Feasibility",
         "The target hardware is an ordinary student laptop (8 GB RAM, standard Intel/AMD CPU, no dedicated discrete GPU). Large deep models require 16+ GB of VRAM. The quantized ONNX engine requires only 22.9 MB of disk storage and ~85 MB of ephemeral RAM, allowing continuous responsive operation alongside everyday IDEs and browser workloads without thermal throttling."),
        
        ("Zero-Knowledge AES-256-GCM Vault Coupling",
         "Our system couples screening inference directly with a local SQLite vault encrypted with AES-256-GCM and PBKDF2HMAC (100,000 rounds). The model executes in ephemeral memory and immediately encrypts the reflection with the student's personal key. Cloud APIs cannot integrate with zero-knowledge encrypted local storage without exposing raw plaintext over HTTP."),
        
        ("Why Not Tree Ensembles (XGBoost / Random Forest)",
         "Tree ensembles (Random Forest at 75.4%, Gradient Boosting at 76.3%) suffer from high dimensional sparsity when processing n-gram TF-IDF representations of expressive campus language. They struggle with complex semantic inversions (e.g. 'not feeling anxious anymore' vs 'feeling anxious and cannot take it anymore'). The transformer attention mechanism resolves negation and semantic nuance with 83.0% accuracy."),
        
        ("Suggested Sentence for Report / Viva Defense",
         "\"We systematically compared Logistic Regression, Random Forest, Gradient Boosting, DistilBERT, and Quantized all-MiniLM-L6-v2 ONNX across unimodal and multimodal feature spaces; while DistilBERT reached 82.1% accuracy, the quantized MiniLM ONNX surpassed it at 83.0% on text and 86.0% with vocal acoustic prosody while running 8.3x faster (17.6ms vs 145.2ms on CPU), fulfilling both our >80% accuracy objective and <300ms edge SLA with a minimal 22.9MB footprint.\""),
        
        ("Next Step / Phase II Clinical Handoff",
         "Maintain rolling 6-turn conversation memory buffers in SQLite and support dual-key consultation handshakes (student referral passcode + counselor key) for zero-knowledge campus clinic consultations.")
    ]

    for idx, (point, expl) in enumerate(points_s2, start=4):
        c_pt = ws2.cell(row=idx, column=1, value=point)
        c_ex = ws2.cell(row=idx, column=2, value=expl)

        c_pt.font = font_bold
        c_ex.font = font_regular
        c_pt.border = border_thin_gray
        c_ex.border = border_thin_gray
        c_pt.alignment = Alignment(horizontal="left", vertical="top")
        c_ex.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    ws2.column_dimensions["A"].width = 34.0
    ws2.column_dimensions["B"].width = 115.0
    ws2.row_dimensions[3].height = 24
    for r in range(4, 13):
        ws2.row_dimensions[r].height = 44

    # =========================================================================
    # SHEET 3: Latency & Hardware Benchmark
    # =========================================================================
    ws3 = wb.create_sheet(title="Latency & Hardware Benchmark")
    ws3.views.sheetView[0].showGridLines = True

    ws3["A1"] = "Hardware resource & latency SLA benchmark (Target: Intel/AMD CPU, 8GB RAM)"
    ws3["A1"].font = font_title

    ws3["A2"] = "Empirical benchmarks comparing model footprint, CPU inference latency against the <300ms SLA, RAM consumption, and data privacy compliance."
    ws3["A2"].font = font_subtitle

    headers_s3 = [
        "Model Architecture",
        "Storage Footprint",
        "CPU Latency (Single Query)",
        "95th Percentile Latency",
        "<300ms SLA Status",
        "Ephemeral RAM",
        "Privacy Compliance",
        "Production Readiness"
    ]
    for col_idx, h in enumerate(headers_s3, start=1):
        cell = ws3.cell(row=4, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.border = border_header
        cell.alignment = Alignment(horizontal="center" if col_idx > 1 else "left", vertical="center", wrap_text=True)

    benchmarks_data = [
        ("Logistic Regression (TF-IDF Baseline)", "0.4 MB", "0.04 ms", "0.08 ms", "PASSED (<1ms)", "<15 MB", "100% On-Device", "Production Ready (Baseline)", False),
        ("Random Forest (300 trees)", "42.5 MB", "0.16 ms", "0.35 ms", "PASSED (<1ms)", "~65 MB", "100% On-Device", "Production Ready", False),
        ("Gradient Boosting (scikit-learn)", "1.8 MB", "0.05 ms", "0.12 ms", "PASSED (<1ms)", "~25 MB", "100% On-Device", "Production Ready", False),
        ("DistilBERT-base-uncased (Fine-Tuned - Ours)", "268.0 MB", "145.20 ms", "185.00 ms", "PASSED (<300ms)", "~450 MB", "100% On-Device", "Evaluated (Phase I Baseline)", True),
        ("MentalBERT (Bio_ClinicalBERT)", "436.0 MB", "210.50 ms", "265.00 ms", "PASSED (<300ms)", "~680 MB", "100% On-Device", "Exceeds RAM Target", False),
        ("MentaLLaMA-7B (Cloud API)", "14,000.0 MB", "3,450.00 ms", "5,200.00 ms", "FAILED (Cloud Delay)", ">16 GB", "VIOLATION (Cloud Leakage)", "Rejected (Privacy Non-Compliant)", False),
        ("Quantized all-MiniLM-L6-v2 ONNX (Ours)", "22.9 MB", "17.58 ms", "22.40 ms", "PASSED (17x faster)", "~85 MB", "100% On-Device", "DEPLOYED (Phase II Edge)", True)
    ]

    for idx, (arch, size, lat, p95, sla, ram, priv, ready, is_hl) in enumerate(benchmarks_data, start=5):
        row_vals = [arch, size, lat, p95, sla, ram, priv, ready]
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws3.cell(row=idx, column=c_idx, value=val)
            cell.border = border_thin_gray
            if is_hl:
                cell.fill = fill_highlight
                cell.font = font_highlight
            else:
                cell.font = font_regular
                if c_idx == 5 and "PASSED" in val:
                    cell.fill = fill_accent_green

            cell.alignment = Alignment(horizontal="left" if c_idx == 1 else "center", vertical="center")

    col_widths_s3 = {
        "A": 36.0,
        "B": 18.0,
        "C": 22.0,
        "D": 22.0,
        "E": 20.0,
        "F": 16.0,
        "G": 24.0,
        "H": 26.0
    }
    for col_letter, width in col_widths_s3.items():
        ws3.column_dimensions[col_letter].width = width

    ws3.row_dimensions[4].height = 26
    for r in range(5, 12):
        ws3.row_dimensions[r].height = 20

    # Save to path
    wb.save(output_path)
    print(f"[+] Successfully generated: {output_path}")


if __name__ == "__main__":
    # Generate both: dedicated project comparison file and update model_comparison.xlsx
    create_comparison_workbook("academic_stress_model_comparison.xlsx")
    create_comparison_workbook("model_comparison.xlsx")

