import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Import the sequential consensus implementation
from modules.sequential_consensus_implementation import (
    categorize_document_with_sequential_consensus,
    review_categorization,
    arbitrate_categorization,
    parse_review_response,
    parse_arbitration_response,
    calculate_agreement_confidence,
    calculate_arbitration_confidence
)

# Import utility functions from the new utils file
from modules.document_categorization_utils import (
    categorize_document,
    categorize_document_detailed,
    combine_categorization_results,
    extract_document_features,
    calculate_multi_factor_confidence,
    apply_confidence_calibration,
    apply_confidence_thresholds
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- UI and Integration Logic ---

def document_categorization():
    """
    Enhanced document categorization with improved confidence metrics and user-defined types.
    Includes Standard, Parallel Consensus, and Sequential Consensus modes.
    """
    st.title("Document Categorization")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_cat"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Initialize session state variables if they don't exist
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": {},
            "errors": {}
        }
    
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
        
    if "document_types" not in st.session_state or not isinstance(st.session_state.document_types, list) or \
       not all(isinstance(item, dict) and "name" in item and "description" in item for item in st.session_state.document_types):
        logger.warning("Initializing/Resetting document_types in session state to default structure.")
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoices", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
            {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
            {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity (e.g., balance sheets, income statements)."},
            {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment between an employer and employee."},
            {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
            {"name": "Other", "description": "Any document not fitting into the specific categories above."}
        ]
    
    num_files = len(st.session_state.selected_files)
    st.write(f"Ready to categorize {num_files} files using Box AI.")
    
    tab1, tab2 = st.tabs(["Categorization", "Settings"])
    
    with tab1:
        # --- Model Selection UI (Common for all modes) ---
        all_models_with_desc = {
            "azure__openai__gpt_4_1_mini": "Azure OpenAI GPT-4.1 Mini: Lightweight multimodal model (Default for Box AI for Docs/Notes Q&A)",
            "google__gemini_2_0_flash_lite_preview": "Google Gemini 2.0 Flash Lite: Lightweight multimodal model (Preview)",
            "azure__openai__gpt_4o_mini": "Azure OpenAI GPT-4o Mini: Lightweight multimodal model",
            "azure__openai__gpt_4o": "Azure OpenAI GPT-4o: Highly efficient multimodal model for complex tasks",
            "azure__openai__gpt_4_1": "Azure OpenAI GPT-4.1: Highly efficient multimodal model for complex tasks",
            "azure__openai__gpt_o3": "Azure OpenAI GPT o3: Highly efficient multimodal model for complex tasks",
            "azure__openai__gpt_o4-mini": "Azure OpenAI GPT o4-mini: Highly efficient multimodal model for complex tasks",
            "google__gemini_2_5_pro_preview": "Google Gemini 2.5 Pro: Optimal for high-volume, high-frequency tasks (Preview)",
            "google__gemini_2_5_flash_preview": "Google Gemini 2.5 Flash: Optimal for high-volume, high-frequency tasks (Preview)",
            "google__gemini_2_0_flash_001": "Google Gemini 2.0 Flash: Optimal for high-volume, high-frequency tasks",
            "google__gemini_1_5_flash_001": "Google Gemini 1.5 Flash: High volume tasks & latency-sensitive applications",
            "google__gemini_1_5_pro_001": "Google Gemini 1.5 Pro: Foundation model for various multimodal tasks",
            "aws__claude_3_haiku": "AWS Claude 3 Haiku: Tailored for various language tasks",
            "aws__claude_3_sonnet": "AWS Claude 3 Sonnet: Advanced language tasks, comprehension & context handling",
            "aws__claude_3_5_sonnet": "AWS Claude 3.5 Sonnet: Enhanced language understanding and generation",
            "aws__claude_3_7_sonnet": "AWS Claude 3.7 Sonnet: Enhanced language understanding and generation",
            "aws__titan_text_lite": "AWS Titan Text Lite: Advanced language processing, extensive contexts",
            "ibm__llama_3_2_instruct": "IBM Llama 3.2 Instruct: Instruction-tuned text model for dialogue, retrieval, summarization",
            "ibm__llama_3_2_90b_vision_instruct": "IBM Llama 3.2 90B Vision Instruct: Instruction-tuned vision model (From Error Log)",
            "ibm__llama_4_scout": "IBM Llama 4 Scout: Natively multimodal model for text and multimodal experiences",
            "xai__grok_3_beta": "xAI Grok 3: Excels at data extraction, coding, summarization (Beta)",
            "xai__grok_3_mini_beta": "xAI Grok 3 Mini: Lightweight model for logic-based tasks (Beta)"
        }
        allowed_model_names = [
            "azure__openai__gpt_4o_mini", "azure__openai__gpt_4_1", "azure__openai__gpt_4_1_mini",
            "google__gemini_1_5_pro_001", "google__gemini_1_5_flash_001", "google__gemini_2_0_flash_001",
            "google__gemini_2_0_flash_lite_preview", "aws__claude_3_haiku", "aws__claude_3_sonnet",
            "aws__claude_3_5_sonnet", "aws__claude_3_7_sonnet", "aws__titan_text_lite",
            "ibm__llama_3_2_90b_vision_instruct", "ibm__llama_4_scout"
        ]
        ai_models_with_desc = {name: all_models_with_desc.get(name, f"{name} (Description not found)")
                               for name in allowed_model_names if name in all_models_with_desc}
        for name in allowed_model_names:
            if name not in ai_models_with_desc:
                 ai_models_with_desc[name] = f"{name} (Description not found)"
                 logger.warning(f"Model 	\'{name}\' from allowed list was missing description, added placeholder.")
        ai_model_names = list(ai_models_with_desc.keys())
        ai_model_options = list(ai_models_with_desc.values())
        
        # Initialize session state variables for models if they don't exist
        if "categorization_ai_model" not in st.session_state:
            st.session_state.categorization_ai_model = ai_model_names[0]
        if "categorization_consensus_mode" not in st.session_state:
            st.session_state.categorization_consensus_mode = "standard"  # Default to standard mode
        if "categorization_model1" not in st.session_state:
            st.session_state.categorization_model1 = ai_model_names[0]
        if "categorization_model2" not in st.session_state:
            # Try to select a different model family for model2 if possible
            for model in ai_model_names:
                if ("claude" in model and "openai" in st.session_state.categorization_model1) or \
                   ("gemini" in model and "openai" in st.session_state.categorization_model1) or \
                   ("openai" in model and "claude" in st.session_state.categorization_model1):
                    st.session_state.categorization_model2 = model
                    break
            else:
                # If no different family found, just use the next model in the list
                idx = min(1, len(ai_model_names) - 1)
                st.session_state.categorization_model2 = ai_model_names[idx]
        if "categorization_model3" not in st.session_state:
            # Try to select a third model from yet another family if possible
            for model in ai_model_names:
                if model != st.session_state.categorization_model1 and model != st.session_state.categorization_model2:
                    if (("claude" in model and "openai" not in st.session_state.categorization_model1 and "openai" not in st.session_state.categorization_model2) or
                        ("gemini" in model and "openai" not in st.session_state.categorization_model1 and "openai" not in st.session_state.categorization_model2) or
                        ("openai" in model and "claude" not in st.session_state.categorization_model1 and "claude" not in st.session_state.categorization_model2)):
                        st.session_state.categorization_model3 = model
                        break
            else:
                # If no different family found, just use another model in the list
                idx = min(2, len(ai_model_names) - 1)
                st.session_state.categorization_model3 = ai_model_names[idx]
        
        st.write("### AI Model Selection")
        
        # Consensus mode selection
        consensus_mode = st.radio(
            "Consensus Mode",
            options=["Standard", "Parallel Consensus", "Sequential Consensus"],
            index=0 if st.session_state.categorization_consensus_mode == "standard" else 
                  1 if st.session_state.categorization_consensus_mode == "parallel" else 2,
            key="consensus_mode_radio",
            help="Standard: Single model, Parallel: Multiple models simultaneously, Sequential: Models review each other's work"
        )
        
        # Update session state based on selection
        st.session_state.categorization_consensus_mode = "standard" if consensus_mode == "Standard" else \
                                                        "parallel" if consensus_mode == "Parallel Consensus" else \
                                                        "sequential"
        
        # --- Model Selection UI based on Consensus Mode ---
        if st.session_state.categorization_consensus_mode == "standard":
            # Single model selection for standard mode
            current_model_name = st.session_state.categorization_ai_model
            if current_model_name not in ai_model_names:
                logger.warning(f"Previously selected categorization model 	\'{current_model_name}\' is not allowed. Defaulting to 	\'{ai_model_names[0]}\''.")
                current_model_name = ai_model_names[0]
                st.session_state.categorization_ai_model = current_model_name
            
            try:
                current_model_desc = ai_models_with_desc.get(current_model_name, ai_model_options[0])
                selected_index = ai_model_options.index(current_model_desc)
            except (ValueError, KeyError):
                logger.error(f"Error finding index for categorization model 	\'{current_model_name}\'. Defaulting to first model.")
                selected_index = 0
                current_model_name = ai_model_names[selected_index]
                st.session_state.categorization_ai_model = current_model_name
                
            selected_model_desc = st.selectbox(
                "Select AI Model for Categorization",
                options=ai_model_options,
                index=selected_index,
                key="ai_model_select_cat",
                help="Choose the AI model for categorization. Only models supported by the Q&A endpoint are listed."
            )
            
            selected_model_name = ""
            for name, desc in ai_models_with_desc.items():
                if desc == selected_model_desc:
                    selected_model_name = name
                    break
            st.session_state.categorization_ai_model = selected_model_name
            selected_model = selected_model_name
            
        elif st.session_state.categorization_consensus_mode == "parallel":
            # Multiple model selection for parallel consensus
            st.write("Select models for parallel consensus:")
            selected_consensus_descs = st.multiselect(
                "Select models for consensus",
                options=ai_model_options,
                default=[
                    ai_models_with_desc.get(st.session_state.categorization_model1, ai_model_options[0]),
                    ai_models_with_desc.get(st.session_state.categorization_model2, ai_model_options[min(1, len(ai_model_options)-1)])
                ],
                help="Select 2-3 models for best results (more models will increase processing time)",
                key="consensus_models_multiselect"
            )
            
            consensus_models = []
            for desc in selected_consensus_descs:
                for name, description in ai_models_with_desc.items():
                    if description == desc:
                        consensus_models.append(name)
                        break
            
            if len(consensus_models) < 1:
                st.warning("Please select at least one model for consensus categorization")
            
            # Update the first two models in session state if available
            if len(consensus_models) >= 1:
                st.session_state.categorization_model1 = consensus_models[0]
            if len(consensus_models) >= 2:
                st.session_state.categorization_model2 = consensus_models[1]
            
            selected_model = st.session_state.categorization_model1  # Default for non-consensus operations
            
        else:  # sequential consensus
            # Sequential model selection with clear roles
            st.write("Select models for sequential consensus:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model 1 selection (Initial Analysis)
                model1_index = 0
                try:
                    model1_desc = ai_models_with_desc.get(st.session_state.categorization_model1, "")
                    if model1_desc in ai_model_options:
                        model1_index = ai_model_options.index(model1_desc)
                except (ValueError, KeyError):
                    model1_index = 0
                
                model1_desc = st.selectbox(
                    "Model 1 (Initial Analysis)",
                    options=ai_model_options,
                    index=model1_index,
                    key="model1_select",
                    help="First model to perform initial document categorization"
                )
                
                for name, desc in ai_models_with_desc.items():
                    if desc == model1_desc:
                        st.session_state.categorization_model1 = name
                        break
            
            with col2:
                # Model 2 selection (Expert Review)
                model2_index = 0
                try:
                    model2_desc = ai_models_with_desc.get(st.session_state.categorization_model2, "")
                    if model2_desc in ai_model_options:
                        model2_index = ai_model_options.index(model2_desc)
                except (ValueError, KeyError):
                    # Try to find a more capable model than Model 1 if possible
                    for i, name in enumerate(ai_model_names):
                        if ("4o" in name and not "mini" in name) or "4_1" in name or "claude_3_5" in name or "claude_3_7" in name:
                            if name != st.session_state.categorization_model1:
                                model2_index = i
                                break
                    else:
                        # If no better model found, just use a different model
                        for i, name in enumerate(ai_model_names):
                            if name != st.session_state.categorization_model1:
                                model2_index = i
                                break
                
                model2_desc = st.selectbox(
                    "Model 2 (Expert Review)",
                    options=ai_model_options,
                    index=model2_index,
                    key="model2_select",
                    help="Second model to review Model 1's categorization (should be more capable than Model 1)"
                )
                
                for name, desc in ai_models_with_desc.items():
                    if desc == model2_desc:
                        st.session_state.categorization_model2 = name
                        break
            
            # Model 3 selection (Arbitration)
            st.write("Model 3 will be used for arbitration only when needed:")
            
            model3_index = 0
            try:
                model3_desc = ai_models_with_desc.get(st.session_state.categorization_model3, "")
                if model3_desc in ai_model_options:
                    model3_index = ai_model_options.index(model3_desc)
            except (ValueError, KeyError):
                # Try to find a model from a different family than Models 1 and 2
                model1_family = ""
                if "openai" in st.session_state.categorization_model1 or "openai" in st.session_state.categorization_model2:
                    model1_family = "openai"
                elif "claude" in st.session_state.categorization_model1 or "claude" in st.session_state.categorization_model2:
                    model1_family = "claude"
                elif "gemini" in st.session_state.categorization_model1 or "gemini" in st.session_state.categorization_model2:
                    model1_family = "gemini"
                
                for i, name in enumerate(ai_model_names):
                    if model1_family == "openai" and "openai" not in name:
                        model3_index = i
                        break
                    elif model1_family == "claude" and "claude" not in name:
                        model3_index = i
                        break
                    elif model1_family == "gemini" and "gemini" not in name:
                        model3_index = i
                        break
                else:
                    # If no different family found, just use another model
                    for i, name in enumerate(ai_model_names):
                        if name != st.session_state.categorization_model1 and name != st.session_state.categorization_model2:
                            model3_index = i
                            break
            
            model3_desc = st.selectbox(
                "Model 3 (Arbitration)",
                options=ai_model_options,
                index=model3_index,
                key="model3_select",
                help="Third model to arbitrate when Models 1 and 2 disagree significantly"
            )
            
            for name, desc in ai_models_with_desc.items():
                if desc == model3_desc:
                    st.session_state.categorization_model3 = name
                    break
            
            # Sequential consensus parameters
            st.write("### Sequential Consensus Parameters")
            
            disagreement_threshold = st.slider(
                "Disagreement Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="disagreement_threshold_slider",
                help="Confidence difference threshold that triggers Model 3 arbitration"
            )
            
            # Store the selected models for use in processing
            selected_model = st.session_state.categorization_model1  # Default for non-consensus operations
        
        # --- Categorization Options and Controls ---
        st.write("### Categorization Options")
        col1_opt, col2_opt = st.columns(2)
        with col1_opt:
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                value=True,
                key="use_two_stage_cat",
                help="When enabled, documents with low confidence will undergo a second analysis",
                disabled=st.session_state.categorization_consensus_mode != "standard"  # Only enable for standard mode
            )
        with col2_opt:
            confidence_threshold = st.slider(
                "Confidence threshold for second-stage",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="confidence_threshold_cat",
                help="Documents with confidence below this threshold will undergo second-stage analysis",
                disabled=not use_two_stage or st.session_state.categorization_consensus_mode != "standard"
            )
        
        col1_ctrl, col2_ctrl = st.columns(2)
        with col1_ctrl:
            start_button = st.button("Start Categorization", key="start_categorization_button_cat", use_container_width=True)
        with col2_ctrl:
            cancel_button = st.button("Cancel Categorization", key="cancel_categorization_button_cat", use_container_width=True)
        
        # --- Processing Logic ---
        if start_button:
            current_doc_types = st.session_state.get("document_types", [])
            valid_categories = [dtype["name"] for dtype in current_doc_types if isinstance(dtype, dict) and "name" in dtype]
            document_types_with_desc = [dtype for dtype in current_doc_types if isinstance(dtype, dict) and "name" in dtype and "description" in dtype]

            if not valid_categories:
                 st.error("Cannot start categorization: No valid document types defined in Settings.")
            else:
                with st.spinner("Categorizing documents..."):
                    st.session_state.document_categorization = {
                        "is_categorized": False,
                        "results": {},
                        "errors": {}
                    }
                    
                    for file in st.session_state.selected_files:
                        file_id = file["id"]
                        file_name = file["name"]
                        
                        try:
                            # Process based on selected consensus mode
                            if st.session_state.categorization_consensus_mode == "sequential":
                                # Sequential consensus processing
                                model1 = st.session_state.categorization_model1
                                model2 = st.session_state.categorization_model2
                                model3 = st.session_state.categorization_model3
                                
                                st.info(f"Processing {file_name} with sequential consensus...")
                                result = categorize_document_with_sequential_consensus(
                                    file_id, 
                                    model1, 
                                    model2, 
                                    model3, 
                                    document_types_with_desc,
                                    disagreement_threshold
                                )
                                
                                # Store the result directly - it already includes multi_factor_confidence and calibrated_confidence
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": result["document_type"],
                                    "confidence": result["confidence"],
                                    "multi_factor_confidence": result.get("multi_factor_confidence", {}),
                                    "calibrated_confidence": result.get("calibrated_confidence", result["confidence"]),
                                    "reasoning": result["reasoning"],
                                    "original_response": result.get("original_response", ""),
                                    "document_features": result.get("document_features", {}),
                                    "sequential_consensus": result.get("sequential_consensus", {}),
                                    "model1_result": result.get("model1_result", {}),
                                    "model2_result": result.get("model2_result", {}),
                                    "model3_result": result.get("model3_result", {})
                                }
                                
                            elif st.session_state.categorization_consensus_mode == "parallel" and len(consensus_models) > 1:
                                # Parallel consensus processing (existing functionality)
                                consensus_results = []
                                model_progress = st.progress(0)
                                model_status = st.empty()
                                for i, model_name_iter in enumerate(consensus_models):
                                    model_status.text(f"Processing with {model_name_iter}...")
                                    result = categorize_document(file_id, model_name_iter, document_types_with_desc)
                                    # Store model name with result for display in reasoning
                                    result["model_name"] = model_name_iter
                                    consensus_results.append(result)
                                    model_progress.progress((i + 1) / len(consensus_models))
                                model_progress.empty()
                                model_status.empty()
                                result = combine_categorization_results(consensus_results, valid_categories, consensus_models)
                                
                                document_features = extract_document_features(file_id)
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    result["confidence"],
                                    document_features,
                                    result["document_type"],
                                    result.get("reasoning", ""),
                                    valid_categories
                                )
                                calibrated_confidence = apply_confidence_calibration(
                                    result["document_type"],
                                    multi_factor_confidence.get("overall", result["confidence"]) 
                                )
                                
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": result["document_type"],
                                    "confidence": result["confidence"],
                                    "multi_factor_confidence": multi_factor_confidence, 
                                    "calibrated_confidence": calibrated_confidence, 
                                    "reasoning": result["reasoning"],
                                    "original_response": result.get("original_response", ""),
                                    "consensus_results": consensus_results,
                                    "document_features": document_features
                                }
                            else:
                                # Standard processing (existing functionality)
                                result = categorize_document(file_id, selected_model, document_types_with_desc)
                                if use_two_stage and result["confidence"] < confidence_threshold:
                                    st.info(f'Low confidence ({result["confidence"]:.2f}) for {file_name}, performing detailed analysis...')
                                    detailed_result = categorize_document_detailed(file_id, selected_model, result["document_type"], document_types_with_desc)
                                    result = {
                                        "document_type": detailed_result["document_type"],
                                        "confidence": detailed_result["confidence"],
                                        "reasoning": detailed_result["reasoning"],
                                        "first_stage_type": result["document_type"],
                                        "first_stage_confidence": result["confidence"],
                                        "original_response": detailed_result.get("original_response", "")
                                    }
                                
                                document_features = extract_document_features(file_id)
                                multi_factor_confidence = calculate_multi_factor_confidence(
                                    result["confidence"],
                                    document_features,
                                    result["document_type"],
                                    result.get("reasoning", ""),
                                    valid_categories
                                )
                                calibrated_confidence = apply_confidence_calibration(
                                    result["document_type"],
                                    multi_factor_confidence.get("overall", result["confidence"]) 
                                )
                                
                                st.session_state.document_categorization["results"][file_id] = {
                                    "file_id": file_id,
                                    "file_name": file_name,
                                    "document_type": result["document_type"],
                                    "confidence": result["confidence"],
                                    "multi_factor_confidence": multi_factor_confidence, 
                                    "calibrated_confidence": calibrated_confidence, 
                                    "reasoning": result["reasoning"],
                                    "original_response": result.get("original_response", ""),
                                    "first_stage_type": result.get("first_stage_type"),
                                    "first_stage_confidence": result.get("first_stage_confidence"),
                                    "document_features": document_features
                                }
                        except Exception as e:
                            logger.error(f"Error categorizing document {file_name}: {str(e)}")
                            st.session_state.document_categorization["errors"][file_id] = {
                                "file_id": file_id,
                                "file_name": file_name,
                                "error": str(e)
                            }
                    
                    st.session_state.document_categorization["results"] = apply_confidence_thresholds(
                        st.session_state.document_categorization["results"]
                    )
                    st.session_state.document_categorization["is_categorized"] = True
                    num_processed = len(st.session_state.document_categorization["results"])
                    num_errors = len(st.session_state.document_categorization["errors"])
                    if num_errors == 0:
                        st.success(f"Categorization complete! Processed {num_processed} files.")
                    else:
                        st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.")
        
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
    if st.button("Add Document Type", key="add_doc_type_button"):
        st.session_state.document_types.append({"name": f"New Type {len(st.session_state.document_types) + 1}", "description": ""})
        st.rerun()

def configure_confidence_thresholds():
    """
    UI for configuring confidence thresholds.
    """
    st.write("Set the confidence score thresholds for automatic acceptance, verification, and rejection.")
    
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    thresholds = st.session_state.confidence_thresholds
    
    # Ensure verification is always less than or equal to auto_accept
    auto_accept = st.slider(
        "Auto-Accept Threshold", 
        min_value=0.0, max_value=1.0, 
        value=thresholds["auto_accept"],
        step=0.01,
        key="auto_accept_slider",
        help="Documents with confidence above this score will be marked as Auto-Accepted."
    )
    
    verification = st.slider(
        "Verification Threshold", 
        min_value=0.0, max_value=auto_accept, # Max is auto_accept
        value=min(thresholds["verification"], auto_accept), # Ensure initial value is valid
        step=0.01,
        key="verification_slider",
        help="Documents with confidence between this score and Auto-Accept will be marked as Needs Verification."
    )
    
    # Ensure rejection is always less than or equal to verification
    rejection = st.slider(
        "Rejection Threshold", 
        min_value=0.0, max_value=verification, # Max is verification
        value=min(thresholds["rejection"], verification), # Ensure initial value is valid
        step=0.01,
        key="rejection_slider",
        help="Documents with confidence between this score and Verification will be marked as Likely Incorrect. Below this score is Low Confidence / Reject."
    )
    
    # Update session state
    st.session_state.confidence_thresholds = {
        "auto_accept": auto_accept,
        "verification": verification,
        "rejection": rejection
    }

def validate_confidence_with_examples():
    """
    Show examples of how confidence scores map to statuses.
    """
    thresholds = st.session_state.confidence_thresholds
    st.write("Example Statuses based on Current Thresholds:")
    
    examples = [0.95, 0.85, 0.75, 0.60, 0.50, 0.40, 0.30]
    data = []
    for conf in examples:
        if conf >= thresholds["auto_accept"]:
            status = "Auto-Accepted"
            color = "green"
        elif conf >= thresholds["verification"]:
            status = "Needs Verification"
            color = "orange"
        elif conf >= thresholds["rejection"]:
            status = "Likely Incorrect"
            color = "red"
        else:
            status = "Low Confidence / Reject"
            color = "darkred"
        data.append({"Confidence": f"{conf:.2f}", "Status": status, "Color": color})
        
    df = pd.DataFrame(data)
    
    def color_status(val):
        color = next((item["Color"] for item in data if item["Status"] == val), "black")
        return f"color: {color}; font-weight: bold;"
        
    st.dataframe(
        df.style.applymap(color_status, subset=["Status"]),
        hide_index=True,
        use_container_width=True
    )

def display_categorization_results():
    """
    Display categorization results with enhanced confidence visualization
    """
    st.write("### Categorization Results")
    results = st.session_state.document_categorization.get("results", {})
    if not results:
        st.info("No categorization results available.")
        return
    
    tab_table, tab_detailed = st.tabs(["Table View", "Detailed View"])
    
    with tab_table:
        results_data = []
        for file_id, result in results.items():
            status = result.get("status", "Review")
            confidence = result.get("calibrated_confidence", result.get("multi_factor_confidence", {}).get("overall", result.get("confidence", 0.0)))
            if confidence >= 0.8: confidence_level, confidence_color = "High", "green"
            elif confidence >= 0.6: confidence_level, confidence_color = "Medium", "orange"
            else: confidence_level, confidence_color = "Low", "red"
            
            consensus_info = ""
            if "sequential_consensus" in result:
                agreement_level = result["sequential_consensus"].get("agreement_level", "")
                consensus_info = f"Sequential: {agreement_level}"
            elif "consensus_results" in result:
                consensus_info = f"Parallel: {len(result['consensus_results'])} models"
            
            results_data.append({
                "File Name": result["file_name"],
                "Document Type": result["document_type"],
                "Confidence": f"{confidence:.2f}",
                "Confidence Level": confidence_level,
                "Confidence Color": confidence_color,
                "Status": status,
                "Consensus": consensus_info
            })
        
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Apply color to confidence level
            def color_confidence(val):
                color = next((item["Confidence Color"] for item in results_data if item["Confidence Level"] == val), "black")
                return f'color: {color}; font-weight: bold'
            
            # Display the table with styled confidence
            st.dataframe(
                df.drop(columns=["Confidence Color"]),
                column_config={
                    "Confidence Level": st.column_config.TextColumn(
                        "Confidence Level",
                        help="Confidence level based on thresholds",
                        width="medium"
                    ),
                    "Confidence": st.column_config.NumberColumn(
                        "Confidence",
                        help="Calculated confidence score",
                        format="%.2f",
                        width="small"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Processing status based on confidence thresholds",
                        width="medium"
                    ),
                    "Consensus": st.column_config.TextColumn(
                        "Consensus",
                        help="Consensus information",
                        width="medium"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
    
    with tab_detailed:
        for file_id, result in results.items():
            with st.expander(f"{result['file_name']} - {result['document_type']}"):
                display_detailed_result(result)

def display_detailed_result(result):
    """
    Display detailed categorization result with enhanced visualization for sequential consensus
    """
    confidence = result.get("calibrated_confidence", result.get("multi_factor_confidence", {}).get("overall", result.get("confidence", 0.0)))
    
    # Display confidence header with appropriate color
    if confidence >= 0.8: confidence_level, confidence_color = "High", "green"
    elif confidence >= 0.6: confidence_level, confidence_color = "Medium", "orange"
    else: confidence_level, confidence_color = "Low", "red"
    
    st.markdown(f"**Document Type:** {result['document_type']}")
    st.markdown(f"**Overall Confidence:** <span style='color:{confidence_color};font-weight:bold'>{confidence_level} ({confidence:.2f})</span>", unsafe_allow_html=True)
    
    # Display sequential consensus information if available
    if "sequential_consensus" in result:
        st.markdown("### Sequential Consensus Details")
        
        agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
        agreement_color = "green" if agreement_level == "Full Agreement" else "orange" if agreement_level == "Partial Agreement" else "blue" if agreement_level == "Arbitrated" else "red"
        
        st.markdown(f"**Agreement Level:** <span style='color:{agreement_color};font-weight:bold'>{agreement_level}</span>", unsafe_allow_html=True)
        
        # Create tabs for each model's results
        model_tabs = []
        if "model1_result" in result:
            model_tabs.append("Model 1 (Initial)")
        if "model2_result" in result:
            model_tabs.append("Model 2 (Review)")
        if "model3_result" in result:
            model_tabs.append("Model 3 (Arbitration)")
        
        if model_tabs:
            tabs = st.tabs(model_tabs)
            
            tab_index = 0
            if "model1_result" in result and tab_index < len(tabs):
                with tabs[tab_index]:
                    model1_result = result["model1_result"]
                    model_name = model1_result.get("model_name", "Unknown")
                    model_confidence = model1_result.get("confidence", 0.0)
                    model_type = model1_result.get("document_type", "Unknown")
                    
                    st.markdown(f"**Model:** {model_name}")
                    st.markdown(f"**Category:** {model_type}")
                    st.markdown(f"**Confidence:** {model_confidence:.2f}")
                    
                    # Display multi-factor confidence if available
                    if "multi_factor_confidence" in model1_result:
                        st.markdown("#### Confidence Factors:")
                        factors = model1_result["multi_factor_confidence"]
                        for factor, value in factors.items():
                            if factor != "overall":
                                st.markdown(f"- {factor.replace('_', ' ').title()}: {value:.2f}")
                    
                    st.markdown("#### Reasoning:")
                    st.markdown(model1_result.get("reasoning", "No reasoning provided"))
                tab_index += 1
            
            if "model2_result" in result and tab_index < len(tabs):
                with tabs[tab_index]:
                    model2_result = result["model2_result"]
                    model_name = model2_result.get("model_name", "Unknown")
                    model_confidence = model2_result.get("confidence", 0.0)
                    model_type = model2_result.get("document_type", "Unknown")
                    
                    st.markdown(f"**Model:** {model_name}")
                    st.markdown(f"**Category:** {model_type}")
                    st.markdown(f"**Confidence:** {model_confidence:.2f}")
                    
                    # Display review assessment if available
                    if "review_assessment" in model2_result:
                        st.markdown("#### Review Assessment:")
                        assessment = model2_result["review_assessment"]
                        st.markdown(f"**Agreement Level:** {assessment.get('agreement_level', 'Unknown')}")
                        st.markdown(f"**Assessment Reasoning:** {assessment.get('assessment_reasoning', 'No assessment provided')}")
                    
                    # Display confidence adjustment factors if available
                    if "confidence_adjustment_factors" in model2_result:
                        st.markdown("#### Confidence Adjustment Factors:")
                        factors = model2_result["confidence_adjustment_factors"]
                        for factor, value in factors.items():
                            st.markdown(f"- {factor.replace('_', ' ').title()}: {value:.2f}")
                    
                    st.markdown("#### Reasoning:")
                    st.markdown(model2_result.get("reasoning", "No reasoning provided"))
                tab_index += 1
            
            if "model3_result" in result and tab_index < len(tabs):
                with tabs[tab_index]:
                    model3_result = result["model3_result"]
                    model_name = model3_result.get("model_name", "Unknown")
                    model_confidence = model3_result.get("confidence", 0.0)
                    model_type = model3_result.get("document_type", "Unknown")
                    
                    st.markdown(f"**Model:** {model_name}")
                    st.markdown(f"**Category:** {model_type}")
                    st.markdown(f"**Confidence:** {model_confidence:.2f}")
                    
                    # Display arbitration assessment if available
                    if "arbitration_assessment" in model3_result:
                        st.markdown("#### Arbitration Assessment:")
                        assessment = model3_result["arbitration_assessment"]
                        st.markdown(f"**Model 1 Assessment:** {assessment.get('model1_assessment', 'No assessment provided')}")
                        st.markdown(f"**Model 2 Assessment:** {assessment.get('model2_assessment', 'No assessment provided')}")
                        st.markdown(f"**Arbitration Reasoning:** {assessment.get('arbitration_reasoning', 'No reasoning provided')}")
                    
                    # Display confidence factors if available
                    if "confidence_factors" in model3_result:
                        st.markdown("#### Confidence Factors:")
                        factors = model3_result["confidence_factors"]
                        for factor, value in factors.items():
                            st.markdown(f"- {factor.replace('_', ' ').title()}: {value:.2f}")
                    
                    st.markdown("#### Reasoning:")
                    st.markdown(model3_result.get("reasoning", "No reasoning provided"))
    
    # Display parallel consensus information if available
    elif "consensus_results" in result:
        st.markdown("### Parallel Consensus Details")
        
        consensus_results = result["consensus_results"]
        st.markdown(f"**Number of Models:** {len(consensus_results)}")
        
        # Create tabs for each model's results
        model_tabs = [f"Model {i+1}" for i in range(len(consensus_results))]
        
        if model_tabs:
            tabs = st.tabs(model_tabs)
            
            for i, (tab, model_result) in enumerate(zip(tabs, consensus_results)):
                with tab:
                    model_name = model_result.get("model_name", "Unknown")
                    model_confidence = model_result.get("confidence", 0.0)
                    model_type = model_result.get("document_type", "Unknown")
                    
                    st.markdown(f"**Model:** {model_name}")
                    st.markdown(f"**Category:** {model_type}")
                    st.markdown(f"**Confidence:** {model_confidence:.2f}")
                    
                    st.markdown("#### Reasoning:")
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
    
    # Display document features
    if "document_features" in result:
        with st.expander("Document Features"):
            features = result["document_features"]
            for key, value in features.items():
                st.markdown(f"**{key}:** {value}")
