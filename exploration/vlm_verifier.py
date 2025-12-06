#!/usr/bin/env python3
"""
VLM-based semantic room verification
Uses GPT-4V or Claude to reason about room suitability
"""

import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import re
from typing import Dict, Optional

class VLMRoomVerifier:
    """
    Verify room type and suitability using Vision-Language Model reasoning
    
    Most intelligent approach - uses actual reasoning instead of heuristics!
    """
    
    def __init__(self, model_type='gpt4v', api_key=None):
        """
        Args:
            model_type: 'gpt4v', 'claude', or 'llava'
            api_key: API key (if using cloud models)
        """
        self.model_type = model_type
        
        # Get API key from parameter or environment
        if api_key is None:
            if model_type == 'gpt4v':
                api_key = os.getenv('OPENAI_API_KEY')
            elif model_type == 'claude':
                api_key = os.getenv('ANTHROPIC_API_KEY')
        
        self.api_key = api_key
        
        # Initialize appropriate client
        if model_type == 'gpt4v':
            if not api_key:
                print("⚠ Warning: OPENAI_API_KEY not set!")
            else:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                print("✓ VLM verifier initialized (GPT-4V)")
        
        elif model_type == 'claude':
            if not api_key:
                print("⚠ Warning: ANTHROPIC_API_KEY not set!")
            else:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                print("✓ VLM verifier initialized (Claude 3.5 Sonnet)")
        
        elif model_type == 'llava':
            # Local model - no API key needed
            self._initialize_llava()
            print("✓ VLM verifier initialized (LLaVA - local)")
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def verify_room(self, target_object: str, env, robot_id) -> Dict:
        """
        🔥 MAIN METHOD: Verify room using VLM reasoning
        
        Args:
            target_object: Query (e.g., "find the bed")
            env: Environment instance
            robot_id: Robot ID
            
        Returns:
            dict with verification results
        """
        
        print("\n" + "="*60)
        print("VLM-BASED ROOM VERIFICATION")
        print("="*60)
        print(f"Target query: {target_object}")
        print(f"Model: {self.model_type}")
        
        # Extract object name
        object_name = self._extract_object_name(target_object)
        print(f"Target object: {object_name}")
        
        # Capture current view
        print("\n📸 Capturing room view...")
        robot_state = env.get_robot_state()
        robot_pos = robot_state['position']
        robot_yaw = robot_state['yaw']
        
        # Get forward-facing view
        rgb, _ = env.get_camera_image_at_angle(
            position=robot_pos,
            yaw=robot_yaw,
            width=640,
            height=480
        )
        
        # Query VLM
        print(f"\n🤖 Querying {self.model_type.upper()} for room analysis...")
        
        vlm_response = self._query_vlm(rgb, object_name)
        
        # Parse response
        room_type = vlm_response.get('room', 'unknown')
        likely = vlm_response.get('likely', 'no')
        confidence = vlm_response.get('confidence', 0.0)
        reasoning = vlm_response.get('reasoning', 'No reasoning provided')
        
        is_promising = (likely.lower() == 'yes') and (confidence > 0.5)
        
        # Display results
        print(f"\n📊 VLM Analysis:")
        print(f"  Room type: {room_type}")
        print(f"  Likely to find {object_name}: {likely}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Reasoning: {reasoning}")
        
        result = {
            'room_type': room_type,
            'likely': likely,
            'confidence': confidence,
            'reasoning': reasoning,
            'is_promising': is_promising,
            'target_object': object_name
        }
        
        print(f"\n{'='*60}")
        if is_promising:
            print(f"✅ VERIFICATION PASSED")
            print(f"   VLM confirms this {room_type} is suitable for {object_name}")
        else:
            print(f"❌ VERIFICATION FAILED")
            print(f"   VLM says this {room_type} is NOT suitable for {object_name}")
        print(f"{'='*60}\n")
        
        return result
    
    def _query_vlm(self, image: np.ndarray, object_name: str) -> Dict:
        """
        Query VLM with image and get structured response
        
        Args:
            image: RGB image (H, W, 3)
            object_name: Target object (e.g., "bed")
            
        Returns:
            Parsed response dict
        """
        
        if self.model_type == 'gpt4v':
            return self._query_gpt4v(image, object_name)
        elif self.model_type == 'claude':
            return self._query_claude(image, object_name)
        elif self.model_type == 'llava':
            return self._query_llava(image, object_name)
        else:
            return self._fallback_response()
    
    def _query_gpt4v(self, image: np.ndarray, object_name: str) -> Dict:
        """Query GPT-4 Vision"""
        
        # Convert to base64
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Construct prompt
        prompt = f"""I am a robot exploring an apartment looking for a {object_name}.

I just reached a new area through a doorway. This is what I see:

[Image shows the view from the doorway]

Please analyze this image and help me decide:

1. What type of room is this? (bedroom, kitchen, bathroom, living room, hallway, office, etc.)

2. Based on what you see, is this the kind of room where you would typically find a {object_name}?

3. Should I proceed to search this room thoroughly for the {object_name}, or should I continue exploring other areas?

4. What do you see that makes you think this?

Respond in this EXACT format:
ROOM: <type of room>
LIKELY: <yes or no>
CONFIDENCE: <0.0 to 1.0>
REASONING: <brief explanation of what you see and why>

Be concise but specific about what objects or features you observe."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # gpt-4o has vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "low"  # Faster, cheaper
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_vlm_response(content)
            
        except Exception as e:
            print(f"  ⚠ GPT-4V query failed: {e}")
            return self._fallback_response()
    
    def _query_claude(self, image: np.ndarray, object_name: str) -> Dict:
        """Query Claude 3.5 Sonnet"""
        
        # Convert to base64
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""I'm a robot looking for a {object_name}. I just reached a new room.

Analyze this image and respond:

ROOM: <room type>
LIKELY: <yes/no - would I find a {object_name} here?>
CONFIDENCE: <0.0-1.0>
REASONING: <what do you see?>

Be specific and concise."""

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            content = message.content[0].text
            return self._parse_vlm_response(content)
            
        except Exception as e:
            print(f"  ⚠ Claude query failed: {e}")
            return self._fallback_response()
    
    def _query_llava(self, image: np.ndarray, object_name: str) -> Dict:
        """Query LLaVA (local model)"""
        
        if not hasattr(self, 'llava_model'):
            print("  ⚠ LLaVA not initialized")
            return self._fallback_response()
        
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        prompt = f"""[INST] <image>
I'm looking for a {object_name}. What type of room is this? Would I find a {object_name} here?

Respond:
ROOM: <type>
LIKELY: <yes/no>
CONFIDENCE: <0.0-1.0>
REASONING: <why> [/INST]"""

        try:
            inputs = self.llava_processor(prompt, pil_image, return_tensors="pt").to(self.llava_model.device)
            
            with torch.no_grad():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.3
                )
            
            response = self.llava_processor.decode(output[0], skip_special_tokens=True)
            return self._parse_vlm_response(response)
            
        except Exception as e:
            print(f"  ⚠ LLaVA query failed: {e}")
            return self._fallback_response()
    
    def _initialize_llava(self):
        """Initialize local LLaVA model"""
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            print("  Loading LLaVA model (this may take a moment)...")
            
            self.llava_processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf"
            )
            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("  ✓ LLaVA loaded successfully")
            
        except Exception as e:
            print(f"  ⚠ Failed to load LLaVA: {e}")
            self.llava_model = None
    
    def _parse_vlm_response(self, text: str) -> Dict:
        """
        Parse VLM response into structured format
        
        Expected format:
        ROOM: bedroom
        LIKELY: yes
        CONFIDENCE: 0.95
        REASONING: This is a bedroom with a bed visible...
        """
        
        result = {
            'room': 'unknown',
            'likely': 'no',
            'confidence': 0.0,
            'reasoning': ''
        }
        
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('ROOM:'):
                result['room'] = line.replace('ROOM:', '').strip()
            
            elif line.startswith('LIKELY:'):
                likely_str = line.replace('LIKELY:', '').strip().lower()
                result['likely'] = 'yes' if 'yes' in likely_str else 'no'
            
            elif line.startswith('CONFIDENCE:'):
                conf_str = line.replace('CONFIDENCE:', '').strip()
                try:
                    result['confidence'] = float(conf_str)
                except:
                    result['confidence'] = 0.5
            
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()
        
        # If reasoning spans multiple lines, capture it
        if 'REASONING:' in text:
            reasoning_start = text.index('REASONING:') + len('REASONING:')
            result['reasoning'] = text[reasoning_start:].strip()
        
        return result
    
    def _fallback_response(self) -> Dict:
        """Fallback response if VLM fails"""
        return {
            'room': 'unknown',
            'likely': 'uncertain',
            'confidence': 0.5,
            'reasoning': 'Unable to analyze room - VLM query failed'
        }
    
    def _extract_object_name(self, query: str) -> str:
        """Extract object name from query"""
        query_lower = query.lower()
        
        # Common objects
        objects = ['bed', 'sofa', 'couch', 'chair', 'table', 'desk', 
                  'fridge', 'refrigerator', 'stove', 'oven', 'microwave',
                  'toilet', 'sink', 'shower', 'bathtub', 'tv', 'television']
        
        for obj in objects:
            if obj in query_lower:
                return obj
        
        # Fallback: last word
        words = query.split()
        return words[-1] if words else query