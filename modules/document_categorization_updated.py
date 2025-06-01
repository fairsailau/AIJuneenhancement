import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Import the sequential consensus implementation
from modules.sequential_consensus_implementation import categorize_document_with_sequential_consensus
from modules.document_categorization_utils import (
    categorize_document,
    extract_document_features,
    calculate_multi_factor_confidence,
    apply_confidence_calibration,
    combine_categorization_results,
    categorize_document_detailed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UPDATED_MODEL_LIST = [
    "azure__openai__gpt_4_1_mini", "azure__openai__gpt_4_1", "azure__openai__gpt_4o_mini", "azure__openai__gpt_4o",
    "google__gemini_2_5_pro_preview", "google__gemini_2_5_flash_preview", "google__gemini_2_0_flash_001", "google__gemini_2_0_flash_lite_preview",
    "aws__claude_3_haiku", "aws__claude_3_sonnet", "aws__claude_3_5_sonnet", "aws__claude_3_7_sonnet",
    "aws__claude_4_sonnet", "aws__claude_4_opus", "aws__titan_text_lite",
    "ibm__llama_3_2_90b_vision_instruct", "ibm__llama_4_scout",
    "xai__grok_3_beta", "xai__grok_3_mini_reasoning_beta", "azure__openai__gpt_o3"
]

def document_categorization():
    """
    Main function for document categorization tab.
    """
    st.title("Document Categorization")
    
    # Initialize session state for document categorization
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": [],
            "errors": []
        }
    
    # Initialize document types if not already in session state
    if "document_types" not in st.session_state:
        logger.warning("Initializing/Resetting document_types in session state to default structure.")
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoice", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
            {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
            {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity."},
            {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment."},
            {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
            {"name": "Other", "description": "Any document not fitting into the specific categories above."}
        ]
    
    # Create tabs for categorization and settings
    tab1, tab2 = st.tabs(["Categorization", "Settings"])
    
    with tab1: # Categorization Tab
        st.write("## AI Model Selection")
        
        # Consensus mode selection
        st.write("### Consensus Mode")
        consensus_mode = st.radio(
            "Consensus Mode",
            ["Standard", "Parallel Consensus", "Sequential Consensus"],
            help="Standard: Single model categorization. Parallel: Multiple models categorize independently. Sequential: Models review each other's work."
        )
        
        if consensus_mode == "Standard":
            # Single model selection
            default_standard_model = "google__gemini_2_0_flash_001"
            if default_standard_model not in UPDATED_MODEL_LIST:
                default_standard_model = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None

            model = st.selectbox(
                "Select AI Model",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_standard_model) if default_standard_model in UPDATED_MODEL_LIST else 0,
                help="Select the AI model to use for document categorization."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        elif consensus_mode == "Parallel Consensus":
            # Multiple model selection
            st.write("Select models for parallel consensus:")
            
            default_parallel_models = [
                m for m in ["google__gemini_2_0_flash_001", "aws__claude_3_sonnet", "azure__openai__gpt_4_1"]
                if m in UPDATED_MODEL_LIST
            ]
            if not default_parallel_models and UPDATED_MODEL_LIST: # Ensure at least one default if possible
                default_parallel_models = [UPDATED_MODEL_LIST[0]]

            models = st.multiselect(
                "Select models for parallel consensus:",
                options=UPDATED_MODEL_LIST,
                default=default_parallel_models
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        else:  # Sequential Consensus
            st.write("### Select models for sequential consensus:")
            
            # Model 1 (Initial Analysis)
            st.write("#### Model 1 (Initial Analysis)")
            default_model1 = "google__gemini_2_0_flash_001"
            if default_model1 not in UPDATED_MODEL_LIST:
                default_model1 = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None
            model1 = st.selectbox(
                "Model 1 (Initial Analysis)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model1) if default_model1 in UPDATED_MODEL_LIST else 0,
                help="This model will perform the initial document categorization."
            )
            
            # Model 2 (Expert Review)
            st.write("#### Model 2 (Expert Review)")
            default_model2 = "aws__claude_3_sonnet"
            if default_model2 not in UPDATED_MODEL_LIST:
                default_model2 = UPDATED_MODEL_LIST[1] if len(UPDATED_MODEL_LIST) > 1 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None)
            model2 = st.selectbox(
                "Model 2 (Expert Review)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model2) if default_model2 in UPDATED_MODEL_LIST else 0,
                help="This model will review Model 1's categorization."
            )
            
            # Model 3 will be used for arbitration only when needed
            st.write("#### Model 3 will be used for arbitration only when needed:")
            
            # Model 3 (Arbitration)
            default_model3 = "aws__claude_3_5_sonnet" # Changed from anthropic__claude_3_5_sonnet
            if default_model3 not in UPDATED_MODEL_LIST:
                default_model3 = UPDATED_MODEL_LIST[2] if len(UPDATED_MODEL_LIST) > 2 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None)
            model3 = st.selectbox(
                "Model 3 (Arbitration)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model3) if default_model3 in UPDATED_MODEL_LIST else 0,
                help="This model will arbitrate if there's significant disagreement between Models 1 and 2."
            )
            
            # Sequential Consensus Parameters
            st.write("### Sequential Consensus Parameters")
            
            # Disagreement threshold
            disagreement_threshold = st.slider(
                "Disagreement Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.01,
                help="Confidence difference threshold that triggers Model 3 arbitration."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage:
                confidence_threshold = st.slider(
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else:
                confidence_threshold = 0.6  # Default value
        
        st.write("## Categorization Options")
        
        # Folder selection
        folder_id = st.text_input("Box Folder ID", value="323454589704")
        
        # Start and cancel buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Categorization")
        with col2:
            cancel_button = st.button("Cancel Categorization")
        
        # Process categorization
        if start_button:
            st.session_state.document_categorization["is_categorized"] = False
            st.session_state.document_categorization["results"] = []
            st.session_state.document_categorization["errors"] = []
            
            # Get folder contents
            try:
                folder = st.session_state.client.folder(folder_id).get()
                items = folder.get_items()
                
                # Filter for files only
                files = [item for item in items if item.type == "file"]

                if not files:
                    st.warning("No files found in the specified folder.")
                else:
                    # Process each file
                    progress_text = st.empty()

                    if consensus_mode == "Standard":
                        progress_text.info(f"Processing {len(files)} files with {model}...")

                        for file in files:
                            try:
                                progress_text.info(f"Processing {file.name}...")
                                
                                if use_two_stage:
                                    result = categorize_document_detailed(
                                        file.id,
                                        model, 
                                        st.session_state.document_types,
                                        confidence_threshold
                                    )
                                else:
                                    result = categorize_document(
                                        file.id,
                                        model, 
                                        st.session_state.document_types
                                    )
                                
                                # Add file info to result
                                result["file_id"] = file.id
                                result["file_name"] = file.name
                                
                                # Add to results
                                st.session_state.document_categorization["results"].append(result)
                                
                            except Exception as e:
                                logger.error(f"Error categorizing document {file.name}: {str(e)}")
                                st.session_state.document_categorization["errors"].append({
                                    "file_id": file.id,
                                    "file_name": file.name,
                                    "error": str(e)
                                })
                    
                    elif consensus_mode == "Parallel Consensus":
                        if not models:
                            st.error("Please select at least one model for parallel consensus.")
                            return

                        progress_text.info(f"Processing {len(files)} files with {len(models)} models in parallel...")

                        for file in files:
                            try:
                                progress_text.info(f"Processing {file.name} with parallel consensus...")
                                
                                # Get results from all selected models
                                model_results = []
                                for model_name in models:
                                    try:
                                        if use_two_stage:
                                            model_result = categorize_document_detailed(
                                                file.id,
                                                model_name,
                                                st.session_state.document_types,
                                                confidence_threshold
                                            )
                                        else:
                                            model_result = categorize_document(
                                                file.id,
                                                model_name,
                                                st.session_state.document_types
                                            )

                                        model_result["model_name"] = model_name
                                        model_results.append(model_result)
                                    except Exception as e:
                                        logger.error(f"Error with model {model_name} for {file.name}: {str(e)}")

                                if model_results:
                                    # Combine results from all models
                                    combined_result = combine_categorization_results(model_results)
                                    
                                    # Add file info to result
                                    combined_result["file_id"] = file.id
                                    combined_result["file_name"] = file.name
                                    combined_result["model_results"] = model_results
                                    
                                    # Add to results
                                    st.session_state.document_categorization["results"].append(combined_result)
                                else:
                                    raise Exception("All models failed to categorize the document")

                            except Exception as e:
                                logger.error(f"Error categorizing document {file.name}: {str(e)}")
                                st.session_state.document_categorization["errors"].append({
                                    "file_id": file.id,
                                    "file_name": file.name,
                                    "error": str(e)
                                })
                    
                    else:  # Sequential Consensus
                        progress_text.info(f"Processing {len(files)} files with sequential consensus...")
                        
                        for file in files:
                            try:
                                progress_text.info(f"Processing {file.name} with sequential consensus...")
                                
                                # Use sequential consensus implementation
                                result = categorize_document_with_sequential_consensus(
                                    file.id,
                                    model1,
                                    model2,
                                    model3,
                                    st.session_state.document_types,
                                    disagreement_threshold
                                )
                                
                                # Add file info to result
                                result["file_id"] = file.id
                                result["file_name"] = file.name
                                
                                # Add to results
                                st.session_state.document_categorization["results"].append(result)
                                
                            except Exception as e:
                                logger.error(f"Error categorizing document {file.name}: {str(e)}")
                                st.session_state.document_categorization["errors"].append({
                                    "file_id": file.id,
                                    "file_name": file.name,
                                    "error": str(e)
                                })
                    
                    # Update status
                    progress_text.empty()
                    st.session_state.document_categorization["is_categorized"] = True
                    num_processed = len(st.session_state.document_categorization["results"])
                    num_errors = len(st.session_state.document_categorization["errors"])
                    if num_errors == 0:
                        st.success(f"Categorization complete! Processed {num_processed} files.")
                    else:
                        st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.")

            except Exception as e:
                st.error(f"Error accessing folder: {str(e)}")

        # --- Results Display ---
        if st.session_state.document_categorization.get("is_categorized", False):
            display_categorization_results()
    
    with tab2: # Settings Tab
        st.write("### Settings")
        st.write("#### Document Types Configuration")
        configure_document_types()

        st.write("#### Confidence Configuration")
        configure_confidence_thresholds()
        with st.expander("Confidence Validation", expanded=False):
            validate_confidence_with_examples()

# --- UI Helper Functions (Settings, Results Display) ---

def configure_document_types():
    """
    UI for configuring document types.
    """
    st.write("Define the categories you want to use for document classification.")
    
    # Use a list of dictionaries for document types
    if "document_types" not in st.session_state:
        st.session_state.document_types = []
    
    # Display existing types
    for i, doc_type in enumerate(st.session_state.document_types):
        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            new_name = st.text_input(f"Name {i+1}", value=doc_type["name"], key=f"doc_type_name_{i}")
        with col2:
            new_desc = st.text_input(f"Description {i+1}", value=doc_type["description"], key=f"doc_type_desc_{i}")
        with col3:
            if st.button("❌", key=f"remove_doc_type_{i}", help="Remove this document type"):
                st.session_state.document_types.pop(i)
                st.rerun()
                
        # Update the dictionary in the list
        st.session_state.document_types[i]["name"] = new_name
        st.session_state.document_types[i]["description"] = new_desc
    
    # Add new type
    st.write("#### Add New Document Type")
    col1, col2, col3 = st.columns([3, 5, 1])
    with col1:
        new_name = st.text_input("Name", key="new_doc_type_name")
    with col2:
        new_desc = st.text_input("Description", key="new_doc_type_desc")
    with col3:
        if st.button("➕", key="add_doc_type", help="Add this document type"):
            if new_name:  # Only add if name is provided
                st.session_state.document_types.append({
                    "name": new_name,
                    "description": new_desc
                })
                st.rerun()

def configure_confidence_thresholds():
    """
    UI for configuring confidence thresholds.
    """
    st.write("Configure confidence thresholds for automatic actions.")
    
    # Initialize confidence thresholds if not in session state
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    # Sliders for thresholds
    st.session_state.confidence_thresholds["auto_accept"] = st.slider(
        "Auto-accept threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["auto_accept"],
        step=0.01,
        help="Documents with confidence above this threshold will be automatically accepted."
    )
    
    st.session_state.confidence_thresholds["verification"] = st.slider(
        "Verification threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["verification"],
        step=0.01,
        help="Documents with confidence above this threshold but below auto-accept will require verification."
    )
    
    st.session_state.confidence_thresholds["rejection"] = st.slider(
        "Rejection threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["rejection"],
        step=0.01,
        help="Documents with confidence below this threshold will be automatically rejected."
    )

def validate_confidence_with_examples():
    """
    UI for validating confidence thresholds with examples.
    """
    st.write("Test confidence thresholds with example values.")
    
    test_confidence = st.slider(
        "Test confidence value",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )
    
    # Determine status based on thresholds
    if test_confidence >= st.session_state.confidence_thresholds["auto_accept"]:
        status = "Auto-Accepted"
        color = "green"
    elif test_confidence >= st.session_state.confidence_thresholds["verification"]:
        status = "Needs Verification"
        color = "orange"
    elif test_confidence >= st.session_state.confidence_thresholds["rejection"]:
        status = "Low Confidence"
        color = "red"
    else:
        status = "Auto-Rejected"
        color = "red"
    
    st.markdown(f"Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

def display_categorization_results():
    """
    Display categorization results in a table.
    """
    st.write("## Categorization Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Create a DataFrame from results
        if st.session_state.document_categorization["results"]:
            data = []
            for result in st.session_state.document_categorization["results"]:
                # Determine status based on confidence
                confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
                
                if confidence >= st.session_state.confidence_thresholds["auto_accept"]:
                    status = "Auto-Accepted"
                elif confidence >= st.session_state.confidence_thresholds["verification"]:
                    status = "Needs Verification"
                elif confidence >= st.session_state.confidence_thresholds["rejection"]:
                    status = "Low Confidence"
                else:
                    status = "Auto-Rejected"
                
                # Get consensus info if available
                consensus_info = ""
                if "model_results" in result:
                    consensus_info = f"Parallel: {len(result['model_results'])} models"
                elif "sequential_consensus" in result:
                    agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
                    consensus_info = f"Sequential: {agreement_level}"
                
                data.append({
                    "File Name": result["file_name"],
                    "Document Type": result["document_type"],
                    "Confidence": round(confidence, 2),
                    "Confidence Level": "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low",
                    "Status": status,
                    "Consensus": consensus_info
                })
            
            df = pd.DataFrame(data)
            
            # Function to color status cells
            def color_status(val):
                if val == "Auto-Accepted":
                    return 'background-color: #c6efce; color: #006100'
                elif val == "Needs Verification":
                    return 'background-color: #ffeb9c; color: #9c5700'
                else:
                    return 'background-color: #ffc7ce; color: #9c0006'
            
            # Display the styled DataFrame
            st.dataframe(
                df.style.map(color_status, subset=["Status"]),
                use_container_width=True
            )
        else:
            st.info("No categorization results available.")
    
    with tab2:
        # Detailed view with selection
        if st.session_state.document_categorization["results"]:
            file_names = [result["file_name"] for result in st.session_state.document_categorization["results"]]
            selected_file = st.selectbox("Select a file to view details", file_names)
            
            # Find the selected result
            selected_result = next(
                (r for r in st.session_state.document_categorization["results"] if r["file_name"] == selected_file),
                None
            )
            
            if selected_result:
                display_detailed_result(selected_result)
        else:
            st.info("No categorization results available.")

def display_detailed_result(result):
    """
    Display detailed categorization result for a single file.
    """
    st.markdown(f"# {result['file_name']} - {result['document_type']}")
    
    # Display overall confidence
    confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
    confidence_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
    st.markdown(f"## Overall Confidence: <span style='color:{confidence_color}'>{'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'Low'} ({confidence:.2f})</span>", unsafe_allow_html=True)
    
    # Display sequential consensus details if available
    if "sequential_consensus" in result:
        st.markdown("## Sequential Consensus Details")
        
        agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
        agreement_color = "green" if agreement_level == "Full Agreement" else "orange" if agreement_level == "Partial Agreement" else "red"
        st.markdown(f"### Agreement Level: <span style='color:{agreement_color}'>{agreement_level}</span>", unsafe_allow_html=True)
        
        # Create tabs for each model
        model_tabs = st.tabs(["Model 1 (Initial)", "Model 2 (Review)", "Model 3 (Arbitration)"])
        
        # Model 1 tab
        with model_tabs[0]:
            if "model1_result" in result:
                model1 = result["model1_result"]
                st.markdown(f"### Model: {model1.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model1.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model1.get('confidence', 0.0):.2f}")

                st.markdown("### Reasoning:")
                st.markdown(model1.get("reasoning", "No reasoning provided"))
        
        # Model 2 tab
        with model_tabs[1]:
            if "model2_result" in result:
                model2_details = result["model2_result"]

                # Check for and display Model 2's independent initial assessment
                if "independent_assessment" in model2_details and isinstance(model2_details["independent_assessment"], dict):
                    independent_assessment = model2_details["independent_assessment"]
                    st.markdown("#### Model 2: Independent Initial Assessment")
                    st.markdown(f"Independent Category: {independent_assessment.get('document_type', 'N/A')}")
                    st.markdown(f"Independent Confidence: {independent_assessment.get('confidence', 0.0):.2f}")
                    st.markdown("Independent Reasoning:")
                    st.markdown(independent_assessment.get('reasoning', 'No reasoning provided'))
                    st.markdown("---") # Visual separator
                    st.markdown("#### Model 2: Final Review Assessment (after seeing Model 1)") # Title for the final review part
                else:
                    # If no independent assessment, still show a title for consistency or fallback
                    st.markdown("#### Model 2: Review Assessment")

                # Display Model 2's final (potentially reviewed) assessment
                st.markdown(f"### Model: {model2_details.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model2_details.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model2_details.get('confidence', 0.0):.2f}")

                st.markdown("### Review Assessment Details:") # Changed title for clarity
                if "review_assessment" in model2_details:
                    st.markdown(f"**Agreement Level:** {model2_details['review_assessment'].get('agreement_level', 'Unknown')}")
                    st.markdown(f"**Assessment Reasoning:** {model2_details['review_assessment'].get('assessment_reasoning', 'No assessment provided')}")

                st.markdown("### Confidence Adjustment Factors:")
                if "confidence_adjustment_factors" in model2_details:
                    factors = model2_details["confidence_adjustment_factors"]
                    st.markdown(f"* Agreement Bonus: {factors.get('agreement_bonus', 0.0):.2f}")
                    st.markdown(f"* Disagreement Penalty: {factors.get('disagreement_penalty', 0.0):.2f}")
                    st.markdown(f"* Reasoning Quality: {factors.get('reasoning_quality', 0.0):.2f}")

                st.markdown("### Final Reasoning:") # Changed title for clarity
                st.markdown(model2_details.get("reasoning", "No reasoning provided"))
        
        # Model 3 tab
        with model_tabs[2]:
            if "model3_result" in result:
                model3 = result["model3_result"]
                st.markdown(f"### Model: {model3.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model3.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model3.get('confidence', 0.0):.2f}")

                if "arbitration_assessment" in model3:
                    st.markdown("### Arbitration Assessment:")
                    st.markdown(f"#### Model 1 Assessment: {model3['arbitration_assessment'].get('model1_assessment', 'No assessment provided')}")
                    st.markdown(f"#### Model 2 Assessment: {model3['arbitration_assessment'].get('model2_assessment', 'No assessment provided')}")
                    st.markdown(f"#### Arbitration Reasoning: {model3['arbitration_assessment'].get('arbitration_reasoning', 'No reasoning provided')}")

                st.markdown("### Reasoning:")
                st.markdown(model3.get("reasoning", "No reasoning provided"))
            else:
                st.markdown("### No arbitration was needed")
                st.markdown("Model 3 was not used because there was sufficient agreement between Models 1 and 2.")
    
    # Display parallel consensus details if available
    elif "model_results" in result:
        st.markdown("## Parallel Consensus Details")

        # Create tabs for each model
        model_names = [model.get("model_name", f"Model {i+1}") for i, model in enumerate(result["model_results"])]
        model_tabs = st.tabs(model_names)

        for i, (tab, model_result) in enumerate(zip(model_tabs, result["model_results"])):
            with tab:
                st.markdown(f"### Model: {model_result.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model_result.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model_result.get('confidence', 0.0):.2f}")

                st.markdown("### Reasoning:")
                st.markdown(model_result.get("reasoning", "No reasoning provided"))
    
    # Standard display for single model results
    else:
        # Display confidence breakdown
        st.markdown("### Confidence Breakdown")

        multi_factor_confidence = result.get("multi_factor_confidence", {})
        if multi_factor_confidence:
            factors = [
                {"factor": "AI Model", "value": multi_factor_confidence.get("ai_reported", 0.0)},
                {"factor": "Response Quality", "value": multi_factor_confidence.get("response_quality", 0.0)},
                {"factor": "Category Specificity", "value": multi_factor_confidence.get("category_specificity", 0.0)},
                {"factor": "Reasoning Quality", "value": multi_factor_confidence.get("reasoning_quality", 0.0)},
                {"factor": "Document Features Match", "value": multi_factor_confidence.get("document_features_match", 0.0)}
            ]

            for factor in factors:
                factor_value = factor["value"]
                factor_color = "green" if factor_value >= 0.8 else "orange" if factor_value >= 0.6 else "red"
                st.markdown(f"{factor['factor']}: <span style='color:{factor_color}'>{factor_value:.2f}</span>", unsafe_allow_html=True)
                # Create a progress bar
                st.progress(factor_value)
        
        # Display two-stage information if available
        if result.get("first_stage_type"):
            st.markdown("### Two-Stage Analysis")
            st.markdown(f"**First Stage Category:** {result['first_stage_type']}")
            st.markdown(f"**First Stage Confidence:** {result.get('first_stage_confidence', 0.0):.2f}")
            st.markdown("**Second Stage Analysis:** Performed due to low initial confidence")
    
    # Common display elements for all result types
    st.markdown("### Reasoning")
    st.markdown(result.get("reasoning", "No reasoning provided"))
    
    # Display document features in a separate section
    if "document_features" in result:
        st.markdown("### Document Features")
        features = result["document_features"]
        for key, value in features.items():
            st.markdown(f"**{key}:** {value}")
