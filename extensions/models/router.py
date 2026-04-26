"""
Lightweight Model Router

Simple keyword-based model selection.
If user_choice is provided, use it.
Otherwise, detect keywords in prompt.
If nothing matches, return None (system uses default).

Usage:
    model_id = select_model(prompt, user_choice=None)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# Keyword Mappings
# =============================================================================

# Keywords that map to specific model categories
MODEL_KEYWORDS = {
    # Anime/Stylized keywords
    "anime": ["anything", "anime", "manga", "cartoon", "illustration", "2d", "animated style"],
    "stylized": ["painting", "artistic", "oil painting", "watercolor", "digital art", "concept art"],
    
    # Realistic keywords
    "realistic": ["photorealistic", "photo realistic", "realistic", "portrait", "real photo", "cinematic"],
    "dreamshaper": ["dreamshaper", "dream shaper"],
    
    # Portrait/Character
    "portrait": ["portrait", "face", "person", "human", "character"],
    
    # Landscape/Environment
    "landscape": ["landscape", "scenery", "nature", "environment", "outdoor"],
    
    # Fantasy/Sci-fi
    "fantasy": ["fantasy", "magic", "dragon", "medieval", "castle"],
    "scifi": ["sci-fi", "sci fi", "space", "robot", "future", "cyberpunk"],
    
    # Specific models
    "sdxl": ["sdxl", "xl", "stable diffusion xl"],
    "sd15": ["sd 1.5", "sd15", "stable diffusion 1"],
    "sd21": ["sd 2.1", "sd21", "stable diffusion 2"],
    
    # Motion keywords
    "motion": ["animate", "motion", "moving", "video", "gif"],
    "animate": ["animate", "animation"],
}


# =============================================================================
# Model Recommendations per Category
# =============================================================================

MODEL_RECOMMENDATIONS = {
    "anime": "anything_v5",
    "stylized": "solarmix",
    "realistic": "dreamshaper",
    "dreamshaper": "dreamshaper",
    "portrait": "dreamshaper",
    "landscape": "solarmix",
    "fantasy": "dreamshaper",
    "scifi": "dreamshaper",
    "sdxl": "sdxl_base",
    "sd15": "sd15_base",
    "sd21": "sd21_base",
}


# =============================================================================
# Router Result
# =============================================================================

@dataclass
class RouterResult:
    """Result of model selection."""
    model_id: Optional[str]
    reason: str
    confidence: float  # 0.0 to 1.0
    keywords_found: List[str]


# =============================================================================
# Main Router Function
# =============================================================================

def select_model(
    prompt: str,
    user_choice: Optional[str] = None,
) -> Optional[str]:
    """
    Select model based on prompt keywords or user preference.
    
    Args:
        prompt: The text prompt
        user_choice: User-specified model (takes priority)
        
    Returns:
        Model ID or None (system uses default)
    """
    # If user explicitly chose a model, use it
    if user_choice:
        return user_choice
    
    # No prompt to analyze
    if not prompt:
        return None
    
    # Analyze prompt
    result = _analyze_prompt(prompt)
    
    if result.model_id:
        return result.model_id
    
    # No match - return None so system uses default
    return None


def _analyze_prompt(prompt: str) -> RouterResult:
    """Analyze prompt for model keywords."""
    prompt_lower = prompt.lower()
    keywords_found = []
    category_scores: Dict[str, float] = {}
    
    # Check each keyword category
    for category, keywords in MODEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                keywords_found.append(keyword)
                
                # Score this category
                if category not in category_scores:
                    category_scores[category] = 0.0
                category_scores[category] += 1.0
    
    if not category_scores:
        return RouterResult(
            model_id=None,
            reason="No keywords detected",
            confidence=0.0,
            keywords_found=[]
        )
    
    # Get best matching category
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]
    
    # Get recommended model for this category
    model_id = MODEL_RECOMMENDATIONS.get(best_category)
    
    if model_id:
        confidence = min(best_score / 3.0, 1.0)  # Normalize
        return RouterResult(
            model_id=model_id,
            reason=f"Matched category: {best_category}",
            confidence=confidence,
            keywords_found=keywords_found
        )
    
    return RouterResult(
        model_id=None,
        reason="No model recommendation for category",
        confidence=0.0,
        keywords_found=keywords_found
    )


def get_model_recommendations(prompt: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Get multiple model recommendations for a prompt.
    
    Returns ranked list of potential models.
    """
    prompt_lower = prompt.lower()
    results = []
    
    for category, keywords in MODEL_KEYWORDS.items():
        score = 0
        matched = []
        
        for keyword in keywords:
            if keyword in prompt_lower:
                score += 1
                matched.append(keyword)
        
        if score > 0:
            model_id = MODEL_RECOMMENDATIONS.get(category)
            if model_id:
                results.append({
                    "model_id": model_id,
                    "category": category,
                    "score": score,
                    "keywords": matched
                })
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_n]


# =============================================================================
# Convenience
# =============================================================================

def is_motion_prompt(prompt: str) -> bool:
    """Check if prompt requests motion/animation."""
    motion_keywords = ["animate", "animation", "motion", "moving", "video", "gif"]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in motion_keywords)


def suggest_motion_model() -> str:
    """Suggest best available motion model."""
    # Will check what's available
    return "svd"  # Default to SVD


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    test_prompts = [
        "A beautiful anime girl with blue hair",
        "Photorealistic portrait of a person",
        "Fantasy castle on a mountain",
        "Oil painting of a landscape at sunset",
        "A robot in a cyberpunk city",
        "Just a simple landscape",
    ]
    
    print("=== Model Router Tests ===\n")
    
    for prompt in test_prompts:
        result = select_model(prompt)
        recommendations = get_model_recommendations(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Selected: {result}")
        print(f"Recommendations: {[r['model_id'] for r in recommendations]}")
        print(f"Is motion: {is_motion_prompt(prompt)}")
        print()