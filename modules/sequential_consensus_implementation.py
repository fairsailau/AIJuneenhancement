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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def categorize_document_with_sequential_consensus(
    file_id: str, 
    model1: str, 
    model2: str, 
    model3: str, 
    document_types_with_desc: List[Dict[str, str]],
    disagreement_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Perform document categorization using sequential consensus approach.
    
    Args:
        file_id: Box file ID
        model1: First AI model for initial categorization
        model2: Second AI model for review
        model3: Third AI model for arbitration (used only if needed)
        document_types_with_desc: List of document types with descriptions
        disagreement_threshold: Threshold for significant disagreement
        
    Returns:
        Dictionary with categorization results
    """
    logger.info(f"Starting sequential consensus categorization for file {file_id}")
    
    # Phase 1: Initial Analysis (Model 1)
    model1_status = st.empty()
    model1_status.text(f"Phase 1: Initial analysis with {model1}...")
    
    from modules.document_categorization import categorize_document, extract_document_features, calculate_multi_factor_confidence, apply_confidence_calibration
    
    model1_result = categorize_document(file_id, model1, document_types_with_desc)
    
    document_features = extract_document_features(file_id)
    model1_multi_factor_confidence = calculate_multi_factor_confidence(
        model1_result["confidence"],
        document_features,
        model1_result["document_type"],
        model1_result.get("reasoning", ""),
        [dtype["name"] for dtype in document_types_with_desc]
    )
    
    model1_result["multi_factor_confidence"] = model1_multi_factor_confidence
    model1_result["model_name"] = model1
    
    model1_status.text(f"Phase 1 complete: {model1_result['document_type']} ({model1_multi_factor_confidence['overall']:.2f})")
    
    # Phase 2: Expert Review (Model 2)
    model2_status = st.empty()
    model2_status.text(f"Phase 2: Expert review with {model2}...")
    
    model2_result = review_categorization(
        file_id, 
        model2, 
        model1_result, 
        document_types_with_desc
    )
    
    model2_result["model_name"] = model2
    
    # Calculate agreement level
    category_agreement = model1_result["document_type"] == model2_result["document_type"]
    confidence_diff = abs(model1_multi_factor_confidence["overall"] - model2_result["confidence"])
    significant_disagreement = (not category_agreement) or (confidence_diff >= disagreement_threshold)
    
    model2_status.text(f"Phase 2 complete: {model2_result['document_type']} ({model2_result['confidence']:.2f})")
    
    # Phase 3: Arbitration (Model 3) if needed
    final_result = model2_result.copy()
    final_result["model1_result"] = model1_result
    final_result["model2_result"] = model2_result
    
    if significant_disagreement:
        model3_status = st.empty()
        model3_status.text(f"Phase 3: Arbitration with {model3} due to significant disagreement...")
        
        model3_result = arbitrate_categorization(
            file_id,
            model3,
            model1_result,
            model2_result,
            document_types_with_desc
        )
        
        model3_result["model_name"] = model3
        final_result = model3_result
        final_result["model1_result"] = model1_result
        final_result["model2_result"] = model2_result
        final_result["model3_result"] = model3_result
        
        model3_status.text(f"Phase 3 complete: {model3_result['document_type']} ({model3_result['confidence']:.2f})")
    else:
        # Calculate final confidence based on agreement
        final_confidence = calculate_agreement_confidence(
            model1_multi_factor_confidence["overall"],
            model2_result["confidence"],
            model2_result.get("confidence_adjustment_factors", {}).get("initial_confidence_validity", 0.8)
        )
        
        final_result["confidence"] = final_confidence
        final_result["sequential_consensus"] = {
            "agreement_level": "Full Agreement" if category_agreement else "Partial Agreement",
            "confidence_difference": confidence_diff,
            "final_confidence_calculation": "agreement_based"
        }
    
    # Add document features to final result
    final_result["document_features"] = document_features
    
    # Calculate multi-factor confidence for final result
    final_multi_factor_confidence = calculate_multi_factor_confidence(
        final_result["confidence"],
        document_features,
        final_result["document_type"],
        final_result.get("reasoning", ""),
        [dtype["name"] for dtype in document_types_with_desc]
    )
    
    final_result["multi_factor_confidence"] = final_multi_factor_confidence
    
    # Apply calibration
    calibrated_confidence = apply_confidence_calibration(
        final_result["document_type"],
        final_multi_factor_confidence.get("overall", final_result["confidence"])
    )
    
    final_result["calibrated_confidence"] = calibrated_confidence
    
    logger.info(f"Sequential consensus complete for file {file_id}: {final_result['document_type']} ({calibrated_confidence:.2f})")
    
    return final_result

def review_categorization(
    file_id: str, 
    model: str, 
    model1_result: Dict[str, Any], 
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have Model 2 review the categorization from Model 1.
    
    Args:
        file_id: Box file ID
        model: AI model to use for review
        model1_result: Results from Model 1
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with review results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client for review categorization")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    # Format confidence factors for the prompt
    confidence_factors = model1_result.get("multi_factor_confidence", {})
    confidence_factors_text = "\n".join([
        f"- AI Model Confidence: {confidence_factors.get('ai_reported', 0.0):.2f}",
        f"- Response Quality: {confidence_factors.get('response_quality', 0.0):.2f}",
        f"- Category Specificity: {confidence_factors.get('category_specificity', 0.0):.2f}",
        f"- Reasoning Quality: {confidence_factors.get('reasoning_quality', 0.0):.2f}",
        f"- Document Features Match: {confidence_factors.get('document_features_match', 0.0):.2f}"
    ])
    
    prompt = (
        f"You are an expert document reviewer tasked with evaluating both a document and another AI's categorization of that document.\n\n"
        f"INITIAL CATEGORIZATION BY MODEL 1:\n"
        f"Category: {model1_result['document_type']}\n"
        f"Confidence: {model1_result['confidence']:.2f}\n"
        f"Reasoning: {model1_result['reasoning']}\n\n"
        f"Confidence Factor Breakdown:\n{confidence_factors_text}\n\n"
        f"Your task is to:\n"
        f"1. Carefully review the original document\n"
        f"2. Evaluate Model 1's categorization, confidence, and reasoning\n"
        f"3. Provide your own expert assessment\n\n"
        f"Consider the following categories:\n{category_options_text}\n\n"
        f"Respond in the following format:\n\n"
        f"REVIEW ASSESSMENT:\n"
        f"Agreement Level: [Full Agreement / Partial Agreement / Disagreement]\n"
        f"Assessment Reasoning: [Your detailed assessment of Model 1's analysis, including strengths and weaknesses]\n\n"
        f"FINAL CATEGORIZATION:\n"
        f"Category: [Your final category recommendation]\n"
        f"Confidence: [Your confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization]\n\n"
        f"CONFIDENCE ADJUSTMENT FACTORS:\n"
        f"Initial Confidence Validity: [0.0-1.0] - How valid was Model 1's confidence score\n"
        f"Evidence Strength: [0.0-1.0] - Strength of evidence in the document for your category\n"
        f"Reasoning Quality: [0.0-1.0] - Quality of your reasoning process\n"
        f"Categorization Difficulty: [0.0-1.0] - How challenging this document is to categorize"
    )
    
    logger.info(f"Box AI Review Request Prompt for file {file_id} (model: {model}):\n{prompt}")
    
    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }
    
    try:
        logger.info(f"Making Box AI review call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI review response for {file_id}: {json.dumps(response_data)}")
        
        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_review_response(original_response, valid_categories)
            
            return {
                "document_type": parsed_result["category"],
                "confidence": parsed_result["confidence"],
                "reasoning": parsed_result["reasoning"],
                "original_response": original_response,
                "review_assessment": parsed_result["review_assessment"],
                "confidence_adjustment_factors": parsed_result["confidence_adjustment_factors"]
            }
        else:
            logger.warning(f"No answer field or empty answer in Box AI review response for file {file_id}. Response: {response_data}")
            return {
                "document_type": model1_result["document_type"],
                "confidence": model1_result["confidence"] * 0.9,  # Reduce confidence due to review failure
                "reasoning": "Review model failed to provide an assessment. Using original categorization with reduced confidence.",
                "review_assessment": {
                    "agreement_level": "Review Failed",
                    "assessment_reasoning": "The review model failed to provide an assessment."
                },
                "confidence_adjustment_factors": {
                    "initial_confidence_validity": 0.8,
                    "evidence_strength": 0.5,
                    "reasoning_quality": 0.5,
                    "categorization_difficulty": 0.5
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI review call for file {file_id}: {str(e)}")
        return {
            "document_type": model1_result["document_type"],
            "confidence": model1_result["confidence"] * 0.9,  # Reduce confidence due to review failure
            "reasoning": f"Error during review: {str(e)}. Using original categorization with reduced confidence.",
            "review_assessment": {
                "agreement_level": "Review Failed",
                "assessment_reasoning": f"Error during review: {str(e)}"
            },
            "confidence_adjustment_factors": {
                "initial_confidence_validity": 0.8,
                "evidence_strength": 0.5,
                "reasoning_quality": 0.5,
                "categorization_difficulty": 0.5
            }
        }

def arbitrate_categorization(
    file_id: str,
    model: str,
    model1_result: Dict[str, Any],
    model2_result: Dict[str, Any],
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have Model 3 arbitrate between Model 1 and Model 2 when there's significant disagreement.
    
    Args:
        file_id: Box file ID
        model: AI model to use for arbitration
        model1_result: Results from Model 1
        model2_result: Results from Model 2
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with arbitration results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client for arbitration")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    prompt = (
        f"You are an expert arbitrator tasked with resolving a disagreement between two AI models regarding document categorization.\n\n"
        f"MODEL 1 CATEGORIZATION:\n"
        f"Category: {model1_result['document_type']}\n"
        f"Confidence: {model1_result['confidence']:.2f}\n"
        f"Reasoning: {model1_result['reasoning']}\n\n"
        f"MODEL 2 REVIEW AND CATEGORIZATION:\n"
        f"Category: {model2_result['document_type']}\n"
        f"Confidence: {model2_result['confidence']:.2f}\n"
        f"Reasoning: {model2_result['reasoning']}\n"
        f"Assessment of Model 1: {model2_result.get('review_assessment', {}).get('assessment_reasoning', 'No assessment provided')}\n\n"
        f"Your task is to:\n"
        f"1. Carefully review the original document\n"
        f"2. Evaluate both models' categorizations, confidence scores, and reasoning\n"
        f"3. Make a final determination of the correct category\n\n"
        f"Consider the following categories:\n{category_options_text}\n\n"
        f"Respond in the following format:\n\n"
        f"ARBITRATION ASSESSMENT:\n"
        f"Model 1 Assessment: [Brief assessment of Model 1's categorization]\n"
        f"Model 2 Assessment: [Brief assessment of Model 2's categorization]\n"
        f"Arbitration Reasoning: [Your detailed reasoning for your final decision]\n\n"
        f"FINAL CATEGORIZATION:\n"
        f"Category: [Your final category determination]\n"
        f"Confidence: [Your confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization]\n\n"
        f"CONFIDENCE FACTORS:\n"
        f"Evidence Strength: [0.0-1.0] - Strength of evidence in the document for your category\n"
        f"Reasoning Quality: [0.0-1.0] - Quality of your reasoning process\n"
        f"Categorization Difficulty: [0.0-1.0] - How challenging this document is to categorize\n"
        f"Agreement With Model 1: [0.0-1.0] - How much you agree with Model 1's assessment\n"
        f"Agreement With Model 2: [0.0-1.0] - How much you agree with Model 2's assessment"
    )
    
    logger.info(f"Box AI Arbitration Request Prompt for file {file_id} (model: {model}):\n{prompt}")
    
    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }
    
    try:
        logger.info(f"Making Box AI arbitration call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI arbitration response for {file_id}: {json.dumps(response_data)}")
        
        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_arbitration_response(original_response, valid_categories)
            
            # Calculate final confidence using arbitration formula
            model1_confidence = model1_result["confidence"]
            model2_confidence = model2_result["confidence"]
            model3_confidence = parsed_result["confidence"]
            
            model1_category = model1_result["document_type"]
            model2_category = model2_result["document_type"]
            model3_category = parsed_result["category"]
            final_category = model3_category
            
            final_confidence = calculate_arbitration_confidence(
                model1_confidence, model2_confidence, model3_confidence,
                model1_category, model2_category, model3_category,
                final_category
            )
            
            return {
                "document_type": final_category,
                "confidence": final_confidence,
                "reasoning": parsed_result["reasoning"],
                "original_response": original_response,
                "arbitration_assessment": parsed_result["arbitration_assessment"],
                "confidence_factors": parsed_result["confidence_factors"],
                "sequential_consensus": {
                    "agreement_level": "Arbitrated",
                    "final_confidence_calculation": "arbitration_based"
                }
            }
        else:
            logger.warning(f"No answer field or empty answer in Box AI arbitration response for file {file_id}. Response: {response_data}")
            # Fall back to Model 2's result with reduced confidence
            return {
                "document_type": model2_result["document_type"],
                "confidence": model2_result["confidence"] * 0.9,
                "reasoning": "Arbitration model failed to provide an assessment. Using Model 2's categorization with reduced confidence.",
                "arbitration_assessment": {
                    "model1_assessment": "Arbitration failed",
                    "model2_assessment": "Arbitration failed",
                    "arbitration_reasoning": "The arbitration model failed to provide an assessment."
                },
                "confidence_factors": {
                    "evidence_strength": 0.5,
                    "reasoning_quality": 0.5,
                    "categorization_difficulty": 0.5,
                    "agreement_with_model1": 0.5,
                    "agreement_with_model2": 0.8
                },
                "sequential_consensus": {
                    "agreement_level": "Arbitration Failed",
                    "final_confidence_calculation": "fallback_to_model2"
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI arbitration call for file {file_id}: {str(e)}")
        # Fall back to Model 2's result with reduced confidence
        return {
            "document_type": model2_result["document_type"],
            "confidence": model2_result["confidence"] * 0.9,
            "reasoning": f"Error during arbitration: {str(e)}. Using Model 2's categorization with reduced confidence.",
            "arbitration_assessment": {
                "model1_assessment": "Arbitration failed",
                "model2_assessment": "Arbitration failed",
                "arbitration_reasoning": f"Error during arbitration: {str(e)}"
            },
            "confidence_factors": {
                "evidence_strength": 0.5,
                "reasoning_quality": 0.5,
                "categorization_difficulty": 0.5,
                "agreement_with_model1": 0.5,
                "agreement_with_model2": 0.8
            },
            "sequential_consensus": {
                "agreement_level": "Arbitration Failed",
                "final_confidence_calculation": "fallback_to_model2"
            }
        }

def parse_review_response(response_text: str, valid_categories: List[str]) -> Dict[str, Any]:
    """
    Parse the response from the review model.
    
    Args:
        response_text: The response text from the review model
        valid_categories: List of valid category names
        
    Returns:
        Dictionary with parsed review results
    """
    logger.info(f"Parsing review response: {response_text[:150]}...")
    
    # Default values
    result = {
        "category": "Other",
        "confidence": 0.5,
        "reasoning": "",
        "review_assessment": {
            "agreement_level": "Partial Agreement",
            "assessment_reasoning": ""
        },
        "confidence_adjustment_factors": {
            "initial_confidence_validity": 0.5,
            "evidence_strength": 0.5,
            "reasoning_quality": 0.5,
            "categorization_difficulty": 0.5
        }
    }
    
    try:
        # Split the response into sections
        sections = response_text.split("\n\n")
        current_section = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if section.startswith("REVIEW ASSESSMENT:"):
                current_section = "review_assessment"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if line.startswith("Agreement Level:"):
                        agreement_level = line.split(":", 1)[1].strip()
                        result["review_assessment"]["agreement_level"] = agreement_level
                    elif line.startswith("Assessment Reasoning:"):
                        assessment_reasoning = line.split(":", 1)[1].strip()
                        result["review_assessment"]["assessment_reasoning"] = assessment_reasoning
                    
            elif section.startswith("FINAL CATEGORIZATION:"):
                current_section = "categorization"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if line.startswith("Category:"):
                        category = line.split(":", 1)[1].strip()
                        # Find exact or partial match
                        found_match = False
                        for valid_cat in valid_categories:
                            if valid_cat.lower() == category.lower():
                                result["category"] = valid_cat
                                found_match = True
                                break
                        if not found_match:
                            for valid_cat in valid_categories:
                                if valid_cat.lower() in category.lower() or category.lower() in valid_cat.lower():
                                    result["category"] = valid_cat
                                    found_match = True
                                    break
                        if not found_match and category.lower() == "other":
                            result["category"] = "Other"
                            
                    elif line.startswith("Confidence:"):
                        confidence_str = line.split(":", 1)[1].strip()
                        try:
                            confidence = float(confidence_str)
                            result["confidence"] = max(0.0, min(1.0, confidence))
                        except ValueError:
                            logger.warning(f"Could not parse confidence value: {confidence_str}")
                            
                    elif line.startswith("Reasoning:"):
                        reasoning = line.split(":", 1)[1].strip()
                        result["reasoning"] = reasoning
                        
            elif section.startswith("CONFIDENCE ADJUSTMENT FACTORS:"):
                current_section = "confidence_factors"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if ":" in line:
                        key, value_str = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        
                        # Extract the numeric part before any explanation
                        value_match = re.search(r'([0-9]*\.?[0-9]+)', value_str)
                        if value_match:
                            try:
                                value = float(value_match.group(1))
                                value = max(0.0, min(1.0, value))
                                
                                if "initial_confidence" in key:
                                    result["confidence_adjustment_factors"]["initial_confidence_validity"] = value
                                elif "evidence" in key:
                                    result["confidence_adjustment_factors"]["evidence_strength"] = value
                                elif "reasoning" in key:
                                    result["confidence_adjustment_factors"]["reasoning_quality"] = value
                                elif "difficulty" in key or "challenging" in key:
                                    result["confidence_adjustment_factors"]["categorization_difficulty"] = value
                            except ValueError:
                                logger.warning(f"Could not parse confidence factor value: {value_match.group(1)}")
        
        # If we didn't find a reasoning in the FINAL CATEGORIZATION section, use the assessment reasoning
        if not result["reasoning"] and result["review_assessment"]["assessment_reasoning"]:
            result["reasoning"] = result["review_assessment"]["assessment_reasoning"]
            
        # If we still don't have a reasoning, use the original response
        if not result["reasoning"]:
            result["reasoning"] = response_text
            
    except Exception as e:
        logger.error(f"Error parsing review response: {str(e)}")
        result["reasoning"] = f"Error parsing review response: {str(e)}\n\nOriginal response: {response_text}"
        
    logger.info(f"Parsed review results - Category: {result['category']}, Confidence: {result['confidence']:.2f}")
    return result

def parse_arbitration_response(response_text: str, valid_categories: List[str]) -> Dict[str, Any]:
    """
    Parse the response from the arbitration model.
    
    Args:
        response_text: The response text from the arbitration model
        valid_categories: List of valid category names
        
    Returns:
        Dictionary with parsed arbitration results
    """
    logger.info(f"Parsing arbitration response: {response_text[:150]}...")
    
    # Default values
    result = {
        "category": "Other",
        "confidence": 0.5,
        "reasoning": "",
        "arbitration_assessment": {
            "model1_assessment": "",
            "model2_assessment": "",
            "arbitration_reasoning": ""
        },
        "confidence_factors": {
            "evidence_strength": 0.5,
            "reasoning_quality": 0.5,
            "categorization_difficulty": 0.5,
            "agreement_with_model1": 0.5,
            "agreement_with_model2": 0.5
        }
    }
    
    try:
        # Split the response into sections
        sections = response_text.split("\n\n")
        current_section = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if section.startswith("ARBITRATION ASSESSMENT:"):
                current_section = "arbitration_assessment"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if line.startswith("Model 1 Assessment:"):
                        model1_assessment = line.split(":", 1)[1].strip()
                        result["arbitration_assessment"]["model1_assessment"] = model1_assessment
                    elif line.startswith("Model 2 Assessment:"):
                        model2_assessment = line.split(":", 1)[1].strip()
                        result["arbitration_assessment"]["model2_assessment"] = model2_assessment
                    elif line.startswith("Arbitration Reasoning:"):
                        arbitration_reasoning = line.split(":", 1)[1].strip()
                        result["arbitration_assessment"]["arbitration_reasoning"] = arbitration_reasoning
                    
            elif section.startswith("FINAL CATEGORIZATION:"):
                current_section = "categorization"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if line.startswith("Category:"):
                        category = line.split(":", 1)[1].strip()
                        # Find exact or partial match
                        found_match = False
                        for valid_cat in valid_categories:
                            if valid_cat.lower() == category.lower():
                                result["category"] = valid_cat
                                found_match = True
                                break
                        if not found_match:
                            for valid_cat in valid_categories:
                                if valid_cat.lower() in category.lower() or category.lower() in valid_cat.lower():
                                    result["category"] = valid_cat
                                    found_match = True
                                    break
                        if not found_match and category.lower() == "other":
                            result["category"] = "Other"
                            
                    elif line.startswith("Confidence:"):
                        confidence_str = line.split(":", 1)[1].strip()
                        try:
                            confidence = float(confidence_str)
                            result["confidence"] = max(0.0, min(1.0, confidence))
                        except ValueError:
                            logger.warning(f"Could not parse confidence value: {confidence_str}")
                            
                    elif line.startswith("Reasoning:"):
                        reasoning = line.split(":", 1)[1].strip()
                        result["reasoning"] = reasoning
                        
            elif section.startswith("CONFIDENCE FACTORS:"):
                current_section = "confidence_factors"
                lines = section.split("\n")[1:]  # Skip the header
                for line in lines:
                    if ":" in line:
                        key, value_str = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        
                        # Extract the numeric part before any explanation
                        value_match = re.search(r'([0-9]*\.?[0-9]+)', value_str)
                        if value_match:
                            try:
                                value = float(value_match.group(1))
                                value = max(0.0, min(1.0, value))
                                
                                if "evidence" in key:
                                    result["confidence_factors"]["evidence_strength"] = value
                                elif "reasoning" in key:
                                    result["confidence_factors"]["reasoning_quality"] = value
                                elif "difficulty" in key or "challenging" in key:
                                    result["confidence_factors"]["categorization_difficulty"] = value
                                elif "agreement" in key and "model 1" in key:
                                    result["confidence_factors"]["agreement_with_model1"] = value
                                elif "agreement" in key and "model 2" in key:
                                    result["confidence_factors"]["agreement_with_model2"] = value
                            except ValueError:
                                logger.warning(f"Could not parse confidence factor value: {value_match.group(1)}")
        
        # If we didn't find a reasoning in the FINAL CATEGORIZATION section, use the arbitration reasoning
        if not result["reasoning"] and result["arbitration_assessment"]["arbitration_reasoning"]:
            result["reasoning"] = result["arbitration_assessment"]["arbitration_reasoning"]
            
        # If we still don't have a reasoning, use the original response
        if not result["reasoning"]:
            result["reasoning"] = response_text
            
    except Exception as e:
        logger.error(f"Error parsing arbitration response: {str(e)}")
        result["reasoning"] = f"Error parsing arbitration response: {str(e)}\n\nOriginal response: {response_text}"
        
    logger.info(f"Parsed arbitration results - Category: {result['category']}, Confidence: {result['confidence']:.2f}")
    return result

def calculate_agreement_confidence(model1_confidence: float, model2_confidence: float, model2_validity_assessment: float) -> float:
    """
    Calculate final confidence when models agree on category.
    
    Args:
        model1_confidence: Confidence from Model 1
        model2_confidence: Confidence from Model 2
        model2_validity_assessment: Model 2's assessment of Model 1's confidence validity
        
    Returns:
        Final confidence score
    """
    # Model 2's assessment gets higher weight (0.7 vs 0.3)
    # The validity assessment further adjusts how much of Model 1's confidence to consider
    weighted_model1 = model1_confidence * 0.3 * model2_validity_assessment
    weighted_model2 = model2_confidence * 0.7
    
    # Agreement bonus (up to +0.05)
    agreement_factor = min(1.0, abs(model1_confidence - model2_confidence))
    agreement_bonus = 0.05 * (1 - agreement_factor)
    
    final_confidence = weighted_model1 + weighted_model2 + agreement_bonus
    return min(1.0, final_confidence)  # Cap at 1.0

def calculate_arbitration_confidence(
    model1_confidence: float, 
    model2_confidence: float, 
    model3_confidence: float,
    model1_category: str,
    model2_category: str,
    model3_category: str,
    final_category: str
) -> float:
    """
    Calculate final confidence when arbitration is needed.
    
    Args:
        model1_confidence: Confidence from Model 1
        model2_confidence: Confidence from Model 2
        model3_confidence: Confidence from Model 3
        model1_category: Category from Model 1
        model2_category: Category from Model 2
        model3_category: Category from Model 3
        final_category: Final selected category
        
    Returns:
        Final confidence score
    """
    # Identify which models agreed with the final category
    models_in_agreement = []
    confidences_in_agreement = []
    
    if model1_category == final_category:
        models_in_agreement.append(1)
        confidences_in_agreement.append(model1_confidence)
    
    if model2_category == final_category:
        models_in_agreement.append(2)
        confidences_in_agreement.append(model2_confidence)
    
    if model3_category == final_category:
        models_in_agreement.append(3)
        confidences_in_agreement.append(model3_confidence)
    
    # If no models agreed (rare edge case), use model3 confidence with penalty
    if not models_in_agreement:
        return max(0.1, model3_confidence - 0.2)
    
    # Calculate weighted average of agreeing models
    # Model 3 gets highest weight, then Model 2, then Model 1
    weights = {1: 0.2, 2: 0.3, 3: 0.5}
    
    weighted_sum = 0
    weight_sum = 0
    
    for i, model_num in enumerate(models_in_agreement):
        weighted_sum += confidences_in_agreement[i] * weights[model_num]
        weight_sum += weights[model_num]
    
    return weighted_sum / weight_sum if weight_sum > 0 else model3_confidence
