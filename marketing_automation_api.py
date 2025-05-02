#!/usr/bin/env python3
"""
Marketing Image Generator - Maytapi WhatsApp Version
--------------------------------------------
Production version with Maytapi WhatsApp API integration
Configured for Railway.com deployment
Fixed issues:
- Exclusive gpt-image-1 model usage (no DALL-E fallback)
- Proper user session isolation
- Edit command triggering only for the specific user
- Complete session termination after generate
- Improved image handling
- Fixed repeated welcome message issue
- Fixed audio message handling
"""

import os
import base64
import json
import logging
import time
import traceback
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Also add print statements for important logs to ensure they appear in Railway logs
def log_and_print(level, message):
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "DEBUG":
        logger.debug(message)
    
    # Always print to stdout for Railway logs
    print(f"[{level}] {message}")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store processed messages to prevent duplicates
processed_messages = {}

# Store last message time for rate limiting
last_message_time = {}

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    log_and_print("ERROR", "OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

log_and_print("INFO", f"Using OpenAI API key starting with: {OPENAI_API_KEY[:8]}...")

# Initialize OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
log_and_print("INFO", "OpenAI client initialized with API key")
print(f"OpenAI API Key starts with: {OPENAI_API_KEY[:5]}..." if OPENAI_API_KEY else "OpenAI API Key is missing!")

# Maytapi configuration - from environment variables
MAYTAPI_API_TOKEN = os.getenv('MAYTAPI_API_TOKEN', 'a3da67ed-5fdf-4d80-8424-b3afdaf23e33')
MAYTAPI_PRODUCT_ID = os.getenv('MAYTAPI_PRODUCT_ID', '40345e79-c077-432e-a91f-346451203944')
MAYTAPI_PHONE_ID = os.getenv('MAYTAPI_PHONE_ID', '85920')

# Validate required environment variables
required_vars = {
    'MAYTAPI_API_TOKEN': MAYTAPI_API_TOKEN,
    'MAYTAPI_PRODUCT_ID': MAYTAPI_PRODUCT_ID,
    'MAYTAPI_PHONE_ID': MAYTAPI_PHONE_ID
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    log_and_print("ERROR", f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

log_and_print("INFO", f"Environment variables loaded. Product ID: {MAYTAPI_PRODUCT_ID}, Phone ID: {MAYTAPI_PHONE_ID}")

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details"""
        logger.debug(f"Generating master template with product details: {product_details}")
        return f"""Create a professional marketing poster for a product with these specifications:

1. PRODUCT PRESENTATION:
   - Place the product as the central focal point
   - Remove the original background completely
   - Apply professional product shadows for dimension

2. BRANDING ELEMENTS:
   - Company name: "{product_details.get('company_name', '')}" positioned at the top in an elegant, readable font
   - Product name: "{product_details.get('product_name', '')}" below the product image in bold typography
   - Price: "{product_details.get('price', '')}" displayed in a circular badge, bottom right corner using maroon/red with white text
   - Include a tagline: "{product_details.get('tagline', '')}" in italic font between company name and product image
   - Location/address: "{product_details.get('address', '')}" in smaller font below the product name

3. DESIGN STYLE:
   - Professional marketing aesthetics similar to high-end brands
   - Clean, modern layout with balanced spacing
   - Background: light gradient that complements the product's colors
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template"""
        logger.debug(f"Generating beverage template with product details: {product_details}")
        return f"""Create a premium café marketing poster for a beverage with these specifications:

1. PRODUCT ENHANCEMENT:
   - Center the beverage as the focal point
   - Remove the original background completely
   - Keep whipped cream, garnishes, and toppings visible and appetizing
   - Add subtle steam effects for hot drinks or condensation for cold drinks
   - Enhance beverage colors for visual appeal

2. BRANDING PLACEMENT:
   - Company: "{product_details.get('company_name', '')}" at top in elegant script font
   - Product: "{product_details.get('product_name', '')}" below the product
   - Price: "{product_details.get('price', '')}" in a circular price tag, bottom corner
   - Tagline: "{product_details.get('tagline', '')}" below company name
   - Location: "{product_details.get('address', '')}" in smaller font at bottom

3. AESTHETIC ELEMENTS:
   - Warm background color that complements the drink
   - Subtle coffee beans/relevant ingredients in background
   - Soft vignette effect to focus on the drink
   - Style similar to Starbucks or premium coffee shop marketing

Create a professional marketing poster as described above.
"""

    @staticmethod
    def get_food_template(product_details: Dict) -> str:
        """Generate food-specific prompt template"""
        logger.debug(f"Generating food template with product details: {product_details}")
        return f"""Create an appetizing marketing poster for a food product with these specifications:

1. PRODUCT PRESENTATION:
   - Center the food item as the focal point
   - Remove the original background completely
   - Enhance food textures and colors for appetite appeal
   - Add steam effects for hot items or fresh appearance for cold items
   - Ensure the food looks perfectly prepared

2. BRANDING ELEMENTS:
   - Company: "{product_details.get('company_name', '')}" at top in clean font
   - Product: "{product_details.get('product_name', '')}" below the food in bold font
   - Price: "{product_details.get('price', '')}" in an eye-catching badge
   - Tagline: "{product_details.get('tagline', '')}" emphasizing flavor/freshness
   - Location: "{product_details.get('address', '')}" at bottom in readable font

3. MARKETING ENHANCEMENT:
   - Background complementing the food's colors
   - Subtle ingredients in background for context
   - Professional food photography style
   - Balance all elements for premium restaurant look

Create a professional food marketing poster as described above.
"""

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        logger.info("Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        logger.info(f"ImageGenerator initialized with OpenAI API")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API with gpt-image-1 model"""
        try:
            product_name = product_details.get('product_name', 'product')
            log_and_print("INFO", f"Starting image generation for {product_name} as {product_type}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
                log_and_print("INFO", "Using beverage template")
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
                log_and_print("INFO", "Using food template")
            else:
                prompt = PromptTemplates.get_master_template(product_details)
                log_and_print("INFO", "Using master template")
            
            # Check if product image exists
            if not os.path.exists(product_image_path):
                log_and_print("ERROR", f"Product image not found: {product_image_path}")
                return None
            
            # Try to open image to verify it's a valid image file
            try:
                with Image.open(product_image_path) as img:
                    img_format = img.format
                    img_size = img.size
                    log_and_print("INFO", f"Valid image detected: {img_format}, {img_size}px")
            except Exception as img_error:
                log_and_print("ERROR", f"Invalid image file: {str(img_error)}")
                return None
            
            # Generate the image
            log_and_print("INFO", "Sending image generation request to OpenAI API with gpt-image-1 model")
            print(f"Prompt length: {len(prompt)} characters")
            
            # Add retries for production reliability
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    print(f"[DEBUG] API call attempt {retry+1} to OpenAI")
                    
                    # Open the image file for sending
                    with open(product_image_path, "rb") as image_file:
                        # Use images.edit for generation with image input
                        result = self.client.images.edit(
                            model="gpt-image-1",
                            image=image_file,
                            prompt=prompt,
                            size="1024x1024",
                            n=1,
                            response_format="b64_json",
                            quality="low"  # Use low quality to save credits and lower latency
                        )
                    
                    print(f"[DEBUG] API call successful on attempt {retry+1}")
                    # If successful, break the retry loop
                    break
                except Exception as retry_error:
                    error_message = str(retry_error)
                    logger.warning(f"API call attempt {retry+1} failed: {error_message}")
                    print(f"[WARNING] API call attempt {retry+1} failed: {error_message}")
                    
                    # Check for common API errors
                    if "organization is not verified" in error_message.lower():
                        log_and_print("ERROR", "Your OpenAI organization is not verified. Please verify it to use gpt-image-1.")
                        return None
                    elif "rate limit" in error_message.lower():
                        log_and_print("WARNING", "Rate limit exceeded. Waiting longer before retry...")
                        retry_delay *= 2  # Double the delay for rate limit errors
                    
                    if retry < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        print(f"[INFO] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Last attempt failed, re-raise the exception
                        logger.error("All API call retries failed")
                        print("[ERROR] All API call retries failed")
                        return None
                
            log_and_print("INFO", "OpenAI API request completed")
            
            # Process the result
            print(f"[DEBUG] OpenAI result type: {type(result)}")
            print(f"[DEBUG] OpenAI result has data attribute: {hasattr(result, 'data')}")
            if hasattr(result, 'data'):
                print(f"[DEBUG] result.data length: {len(result.data)}")
                if len(result.data) > 0:
                    print(f"[DEBUG] First item type: {type(result.data[0])}")
                    print(f"[DEBUG] First item has b64_json: {hasattr(result.data[0], 'b64_json')}")
            
            if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                log_and_print("INFO", "Image base64 data received, decoding")
                image_bytes = base64.b64decode(result.data[0].b64_json)
                log_and_print("INFO", "Image decoded successfully")
                print(f"[DEBUG] Decoded image bytes length: {len(image_bytes)}")
            else:
                log_and_print("ERROR", "No image data in OpenAI response")
                print("[ERROR] OpenAI response structure differs from expected format")
                return None
                
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
            output_filename = f"{product_name_safe}_{timestamp}.png"
            output_path = os.path.join("images/output", output_filename)
            print(f"[DEBUG] Image will be saved to: {output_path}")
            
            # Verify output directory exists
            os.makedirs("images/output", exist_ok=True)
            print(f"[DEBUG] Verified output directory exists")
            
            # Convert to image
            log_and_print("INFO", "Creating image from bytes")
            try:
                image = Image.open(BytesIO(image_bytes))
                print(f"[DEBUG] Image opened successfully: {image.format} {image.size}px")
            except Exception as img_error:
                print(f"[ERROR] Failed to open image: {str(img_error)}")
                return None
            
            # Optionally resize for optimization
            if image.size[0] > 1500 or image.size[1] > 1500:
                log_and_print("INFO", f"Resizing image from {image.size} to max 1500px")
                image.thumbnail((1500, 1500), Image.LANCZOS)
                print(f"[DEBUG] Image resized to: {image.size}px")
            
            # Save the image
            log_and_print("INFO", f"Saving image to {output_path}")
            try:
                image.save(output_path, format="PNG", optimize=True)
                print(f"[DEBUG] Image saved successfully as PNG")
            except Exception as save_error:
                print(f"[ERROR] Failed to save image: {str(save_error)}")
                return None
            
            log_and_print("INFO", f"Marketing image saved to {output_path}")
            return output_path
                
        except Exception as e:
            log_and_print("ERROR", f"Error generating marketing image: {str(e)}")
            traceback.print_exc()
            return None

###################
# MAYTAPI WHATSAPP API
###################

class MaytapiClient:
    def __init__(self, product_id: str, api_token: str, phone_id: str):
        log_and_print("INFO", f"Initializing MaytapiClient with product ID: {product_id}, phone ID: {phone_id}")
        self.product_id = product_id
        self.api_token = api_token
        self.phone_id = phone_id
        self.api_base_url = f"https://api.maytapi.com/api/{product_id}"
        self.headers = {
            "x-maytapi-key": self.api_token,
            "Content-Type": "application/json"
        }
        log_and_print("INFO", "MaytapiClient initialized successfully")
        
        # Test connection to ensure credentials are valid
        self._test_connection()
    
    def _test_connection(self):
        """Test the API connection to ensure credentials are valid"""
        try:
            # Checking phone status is a simple API call to verify credentials
            response = self._make_request("GET", f"/{self.phone_id}/status")
            if response.get("success"):
                log_and_print("INFO", "Maytapi connection test successful")
                print(f"[DEBUG] Phone status: {json.dumps(response)}")
            else:
                log_and_print("WARNING", f"Maytapi connection test returned unsuccessful: {json.dumps(response)}")
        except Exception as e:
            log_and_print("WARNING", f"Maytapi connection test failed: {str(e)}")
            # We'll continue anyway, as this is just a test
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a request to the Maytapi API"""
        url = f"{self.api_base_url}{endpoint}"
        log_and_print("INFO", f"Making {method} request to {url}")
        print(f"[DEBUG] Headers: {json.dumps(self.headers)}")
        if data:
            print(f"[DEBUG] Request data: {json.dumps(data)}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            else:
                log_and_print("ERROR", f"Invalid method: {method}")
                return {"success": False, "error": "Invalid method"}
            
            # Log response status
            print(f"[DEBUG] Response status code: {response.status_code}")
            
            # Try to get JSON response
            try:
                result = response.json()
                print(f"[DEBUG] Response JSON: {json.dumps(result)[:500]}...")
            except Exception as json_error:
                log_and_print("ERROR", f"Failed to parse JSON response: {str(json_error)}")
                return {"success": False, "error": f"Invalid JSON response: {response.text}"}
            
            # Check for API errors
            if not result.get("success", False) and "error" in result:
                log_and_print("ERROR", f"API error: {result.get('error')}")
            
            return result
        except Exception as e:
            log_and_print("ERROR", f"Error making request to Maytapi API: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def send_message(self, to_number: str, message: str, typing: bool = True) -> Dict:
        """Send a text message via WhatsApp
        
        Args:
            to_number: The recipient's phone number (with country code, no special chars)
            message: The text message to send
            typing: Whether to show typing indicator before sending
            
        Returns:
            Dict with success status and data
        """
        # Apply rate limiting
        current_time = time.time()
        if to_number in last_message_time and current_time - last_message_time[to_number] < 2:
            log_and_print("INFO", f"Rate limiting message to {to_number} - too soon after last message")
            time.sleep(2 - (current_time - last_message_time[to_number]))
        
        # Clean the phone number - only digits, no special chars
        to_number = self._clean_phone_number(to_number)
        
        data = {
            "to_number": to_number,
            "type": "text",
            "message": message
        }
        
        # Add typing indicator if requested
        if typing:
            data["typing"] = "typing"
            data["duration"] = 2  # 2 seconds of typing
        
        log_and_print("INFO", f"Sending message to {to_number}: {message[:50]}{'...' if len(message) > 50 else ''}")
        result = self._make_request("POST", f"/{self.phone_id}/sendMessage", data)
        
        if result.get("success"):
            log_and_print("INFO", f"Message sent successfully to {to_number}")
            # Update rate limiting
            last_message_time[to_number] = time.time()
            return {"success": True, "data": result.get("data", {})}
        else:
            log_and_print("ERROR", f"Error sending message: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def send_media(self, to_number: str, caption: str = "", media_url: str = None, 
                   media_base64: str = None, filename: str = None) -> Dict:
        """Send a media message via WhatsApp
        
        Args:
            to_number: The recipient's phone number
            caption: Optional text caption for the media
            media_url: URL of the media (either this or media_base64 must be provided)
            media_base64: Base64-encoded media data
            filename: Filename for the media (required for base64)
            
        Returns:
            Dict with success status and data
        """
        # Apply rate limiting
        current_time = time.time()
        if to_number in last_message_time and current_time - last_message_time[to_number] < 2:
            log_and_print("INFO", f"Rate limiting media message to {to_number} - too soon after last message")
            time.sleep(2 - (current_time - last_message_time[to_number]))
        
        # Clean the phone number - only digits, no special chars
        to_number = self._clean_phone_number(to_number)
        
        data = {
            "to_number": to_number,
            "type": "media",
            "text": caption  # Maytapi uses 'text' for caption
        }
        
        # Add either URL or base64 data
        if media_url:
            log_and_print("INFO", f"Sending media from URL to {to_number}")
            data["message"] = media_url
        elif media_base64:
            log_and_print("INFO", f"Sending media from base64 to {to_number}")
            data["message"] = f"data:image/png;base64,{media_base64}"
            # Set filename if provided
            if filename:
                data["filename"] = filename
        else:
            log_and_print("ERROR", "No media URL or base64 data provided")
            return {"success": False, "error": "No media provided"}
        
        result = self._make_request("POST", f"/{self.phone_id}/sendMessage", data)
        
        if result.get("success"):
            log_and_print("INFO", f"Media sent successfully to {to_number}")
            # Update rate limiting
            last_message_time[to_number] = time.time()
            return {"success": True, "data": result.get("data", {})}
        else:
            log_and_print("ERROR", f"Error sending media: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def _clean_phone_number(self, phone_number: str) -> str:
        """Clean a phone number to contain only digits with country code
        
        Args:
            phone_number: The phone number to clean
            
        Returns:
            Cleaned phone number with only digits, preserving country code
        """
        # If already in Maytapi format, return as is
        if '@c.us' in phone_number:
            # Extract just the numbers from Maytapi format
            return phone_number.split('@')[0]
        
        # Remove any non-numeric characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Ensure it has country code (this is simplified, might need more sophistication)
        if len(clean_number) < 10:
            log_and_print("WARNING", f"Phone number too short, might be missing country code: {clean_number}")
        
        return clean_number
    
    def check_phone_number(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp
        
        Args:
            phone_number: The phone number to check
            
        Returns:
            Boolean indicating if the number is registered
        """
        # Clean the phone number
        clean_number = self._clean_phone_number(phone_number)
        
        # Use the checkPhones endpoint
        data = {
            "numbers": [clean_number]
        }
        
        result = self._make_request("POST", f"/{self.phone_id}/checkPhones", data)
        
        if result.get("success") and "data" in result:
            # Check the first number's validity
            numbers_data = result["data"]
            if numbers_data and len(numbers_data) > 0:
                is_valid = numbers_data[0].get("valid", False)
                log_and_print("INFO", f"Phone number {clean_number} is valid on WhatsApp: {is_valid}")
                return is_valid
        
        log_and_print("WARNING", f"Failed to check if phone number is valid: {result.get('error', 'Unknown error')}")
        # Default to True to prevent blocking legitimate numbers if check fails
        return True

###################
# SESSION MANAGER
###################

class SessionManager:
    """Manages user sessions with proper isolation"""
    
    def __init__(self):
        self.sessions = {}
        log_and_print("INFO", "Session Manager initialized")
    
    def get_user_id(self, from_number: str) -> str:
        """Generate a consistent unique user ID from phone number
        
        Args:
            from_number: User's WhatsApp number (e.g. 1234567890@c.us)
            
        Returns:
            Unique user ID
        """
        # Extract just the number part if Maytapi format
        if '@' in from_number:
            number = from_number.split('@')[0]
        else:
            number = from_number
            
        # Clean the number to just digits
        number = ''.join(filter(str.isdigit, number))
        
        # Hash it for consistent unique ID
        return hashlib.md5(number.encode()).hexdigest()
    
    def create_session(self, from_number: str) -> Dict:
        """Create a new session for a user
        
        Args:
            from_number: User's WhatsApp number
            
        Returns:
            New session dictionary
        """
        user_id = self.get_user_id(from_number)
        session = {
            "user_id": user_id,
            "phone_number": from_number,
            "state": "waiting_for_command",
            "product_image": None,
            "details": {},
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "welcomed": False,  # Track if welcome message has been sent
        }
        self.sessions[user_id] = session
        log_and_print("INFO", f"Created new session for user {user_id}")
        return session
    
    def get_session(self, from_number: str, create_if_not_exists: bool = True) -> Dict:
        """Get a user's session
        
        Args:
            from_number: User's WhatsApp number
            create_if_not_exists: Whether to create a session if none exists
            
        Returns:
            User's session dictionary
        """
        user_id = self.get_user_id(from_number)
        
        # Check if session exists
        if user_id in self.sessions:
            # Update last active time
            self.sessions[user_id]["last_active"] = datetime.now().isoformat()
            return self.sessions[user_id]
        
        # Create new session if requested
        if create_if_not_exists:
            return self.create_session(from_number)
        
        return None
    
    def update_session(self, from_number: str, update_data: Dict) -> Dict:
        """Update a user's session with new data
        
        Args:
            from_number: User's WhatsApp number
            update_data: Dictionary of data to update
            
        Returns:
            Updated session dictionary
        """
        session = self.get_session(from_number)
        for key, value in update_data.items():
            session[key] = value
        
        # Update last active time
        session["last_active"] = datetime.now().isoformat()
        
        # Store back to sessions
        user_id = self.get_user_id(from_number)
        self.sessions[user_id] = session
        
        log_and_print("INFO", f"Updated session for user {user_id}")
        return session
    
    def end_session(self, from_number: str) -> bool:
        """End a user's session
        
        Args:
            from_number: User's WhatsApp number
            
        Returns:
            Boolean indicating if session was ended
        """
        user_id = self.get_user_id(from_number)
        
        if user_id in self.sessions:
            # Get existing welcomed state
            welcomed = self.sessions[user_id].get('welcomed', False)
            
            # Reset session to initial state, but keep welcomed state
            self.sessions[user_id] = {
                "user_id": user_id,
                "phone_number": from_number,
                "state": "waiting_for_command",
                "product_image": None,
                "details": {},
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "welcomed": welcomed  # Preserve welcomed status
            }
            log_and_print("INFO", f"Ended session for user {user_id}, preserved welcomed state: {welcomed}")
            return True
        
        return False
    
    def clean_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than specified hours
        
        Args:
            max_age_hours: Maximum session age in hours
            
        Returns:
            Number of sessions cleaned
        """
        now = datetime.now()
        sessions_to_remove = []
        
        for user_id, session in self.sessions.items():
            last_active = datetime.fromisoformat(session["last_active"])
            age_hours = (now - last_active).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_remove.append(user_id)
        
        # Remove the old sessions
        for user_id in sessions_to_remove:
            del self.sessions[user_id]
        
        if sessions_to_remove:
            log_and_print("INFO", f"Cleaned {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)

###################
# IMAGE HANDLER
###################

class ImageHandler:
    """Handles image processing and storage"""
    
    @staticmethod
    def save_image_from_bytes(image_bytes: bytes, user_id: str, filename: str = None) -> Tuple[str, bool]:
        """Save image bytes to disk
        
        Args:
            image_bytes: Raw image data
            user_id: Unique user ID
            filename: Optional filename
            
        Returns:
            Tuple of (file path, success boolean)
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"whatsapp_image_{user_id}_{timestamp}.jpg"
            
            # Ensure input directory exists
            os.makedirs("images/input", exist_ok=True)
            
            # Full path to save image
            image_path = os.path.join("images/input", filename)
            
            # Save bytes directly
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Verify file was saved and is a valid image
            if not os.path.exists(image_path):
                log_and_print("ERROR", f"Failed to save image to {image_path}")
                return None, False
            
            # Try to open as image to validate
            try:
                with Image.open(image_path) as img:
                    img_format = img.format
                    img_size = img.size
                log_and_print("INFO", f"Image saved to {image_path} ({img_format}, {img_size}px)")
                return image_path, True
            except Exception as img_error:
                log_and_print("ERROR", f"Saved file is not a valid image: {str(img_error)}")
                # Remove invalid file
                os.remove(image_path)
                return None, False
                
        except Exception as e:
            log_and_print("ERROR", f"Error saving image: {str(e)}")
            return None, False
    
    @staticmethod
    def extract_image_from_whatsapp(media_data: Any) -> Optional[bytes]:
        """Extract image bytes from WhatsApp message data
        
        Args:
            media_data: Media data from WhatsApp message
            
        Returns:
            Image bytes or None if extraction failed
        """
        try:
            image_bytes = None
            
            # Print media data structure for debugging
            print(f"[DEBUG] Media data type: {type(media_data)}")
            print(f"[DEBUG] Media data: {json.dumps(media_data) if isinstance(media_data, dict) else 'Not a dict'}")
            
            # Try to extract image data from various possible formats
            if media_data:
                # Try direct data field first
                if isinstance(media_data, dict) and 'data' in media_data:
                    log_and_print("INFO", "Found image data in 'data' field")
                    image_bytes = base64.b64decode(media_data['data'])
                
                # Try mimetype field next
                elif isinstance(media_data, dict) and 'mimetype' in media_data and 'body' in media_data:
                    log_and_print("INFO", "Found image data in 'body' field with mimetype")
                    image_bytes = base64.b64decode(media_data['body'])
                
                # Try for Maytapi-specific format - file field
                elif isinstance(media_data, dict) and 'file' in media_data:
                    log_and_print("INFO", "Found image data in Maytapi 'file' field")
                    if isinstance(media_data['file'], str):
                        # Handle base64-encoded data
                        if media_data['file'].startswith('data:'):
                            # Extract base64 part after the comma
                            base64_data = media_data['file'].split(',', 1)[1]
                            image_bytes = base64.b64decode(base64_data)
                        # Handle URL
                        elif media_data['file'].startswith('http'):
                            response = requests.get(media_data['file'])
                            if response.status_code == 200:
                                image_bytes = response.content
                            else:
                                log_and_print("ERROR", f"Failed to download image from URL: {response.status_code}")
                
                # Try media field from Maytapi webhook format
                elif isinstance(media_data, dict) and isinstance(media_data.get('message'), dict):
                    if 'media' in media_data['message']:
                        log_and_print("INFO", "Found image in message.media field")
                        media_url = media_data['message']['media']
                        if isinstance(media_url, str) and media_url.startswith('http'):
                            response = requests.get(media_url)
                            if response.status_code == 200:
                                image_bytes = response.content
                            else:
                                log_and_print("ERROR", f"Failed to download image from message.media URL: {response.status_code}")
            
            return image_bytes
        
        except Exception as e:
            log_and_print("ERROR", f"Error extracting image data: {str(e)}")
            traceback.print_exc()
            return None
    
    @staticmethod
    def create_placeholder_image() -> bytes:
        """Create a simple placeholder image
        
        Returns:
            Image bytes for a placeholder
        """
        try:
            # Create a simple white 300x300 image
            img = Image.new('RGB', (300, 300), color='white')
            
            # Add some text indicating it's a placeholder
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Try to use a system font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except:
                font = ImageFont.load_default()
            
            text = "Placeholder Image"
            
            # Handle different PIL versions for text drawing
            try:
                # For newer PIL versions
                text_width = draw.textlength(text, font=font)
                draw.text(
                    ((300 - text_width) / 2, 140),
                    text,
                    fill='black',
                    font=font
                )
            except AttributeError:
                # For older PIL versions
                text_width, _ = draw.textsize(text, font=font)
                draw.text(
                    ((300 - text_width) / 2, 140),
                    text,
                    fill='black',
                    font=font
                )
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
            
        except Exception as e:
            log_and_print("ERROR", f"Error creating placeholder image: {str(e)}")
            # Return a minimal 1x1 transparent PNG as fallback
            return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, maytapi_product_id: str, maytapi_api_token: str, maytapi_phone_id: str):
        log_and_print("INFO", "Initializing MarketingBot")
        self.image_generator = ImageGenerator(openai_key)
        self.whatsapp_client = MaytapiClient(
            product_id=maytapi_product_id,
            api_token=maytapi_api_token,
            phone_id=maytapi_phone_id
        )
        self.session_manager = SessionManager()
        self.image_handler = ImageHandler()
        log_and_print("INFO", "MarketingBot initialized with all components")
    
    def process_request(self, user_id: str, from_number: str, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Dict:
        """Process a marketing image request
        
        Args:
            user_id: Unique user ID
            from_number: User's WhatsApp number
            product_image_path: Path to product image
            product_details: Dictionary of product details
            product_type: Type of product (beverage, food, etc.)
            
        Returns:
            Dictionary with process results
        """
        try:
            log_and_print("INFO", f"Processing request for user {user_id}")
            print(f"[DEBUG] Product details: {json.dumps(product_details)}")
            print(f"[DEBUG] Product image path: {product_image_path}, exists: {os.path.exists(product_image_path)}")
            
            # Send processing message to user
            self.whatsapp_client.send_message(
                from_number,
                "✨ Generating your marketing image...\n"
                "This may take up to 30 seconds."
            )
            
            # Generate marketing image
            log_and_print("INFO", "Starting image generation")
            output_path = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type
            )
            
            if output_path and os.path.exists(output_path):
                log_and_print("INFO", f"Image generated successfully: {output_path}")
                # Get the Railway app URL for serving images
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
                image_url = f"{app_url}/images/output/{os.path.basename(output_path)}"
                log_and_print("INFO", f"Image URL: {image_url}")
                
                return {
                    "success": True,
                    "image_path": output_path,
                    "image_url": image_url
                }
            else:
                log_and_print("ERROR", "Failed to generate marketing image")
                
                # Send error message to user
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, I couldn't generate the marketing image.\n\n"
                    "This could be due to API limits or issues with the image processing. "
                    "Please try again by sending 'edit'."
                )
                
                return {
                    "success": False,
                    "error": "Failed to generate marketing image"
                }
                
        except Exception as e:
            log_and_print("ERROR", f"Error processing request: {str(e)}")
            traceback.print_exc()
            
            # Send error message to user
            self.whatsapp_client.send_message(
                from_number,
                "Sorry, there was an error processing your request.\n\n"
                "Please try again by sending 'edit'."
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_text_message(self, from_number: str, text: str):
        """Handle incoming text messages
        
        Args:
            from_number: User's WhatsApp number
            text: Message text
        """
        try:
            print(f"[DEBUG] Received text: '{text}' from {from_number}")
            log_and_print("INFO", f"Processing text message: '{text}' from {from_number}")
            
            # Only process private chats (not groups) and only if from_number is valid
            if not from_number or "@g.us" in from_number:
                log_and_print("INFO", f"Message from group or invalid number: {from_number}. Ignored.")
                return
            
            # Get user's session (creates a new one if needed)
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            log_and_print("INFO", f"Current session state for user {user_id}: {session['state']}")
            
            # Check for edit command - this starts a new session
            if text.lower() == 'edit':
                log_and_print("INFO", f"User {user_id} sent 'edit' command")
                
                # Update session to waiting for image
                self.session_manager.update_session(from_number, {
                    "state": "waiting_for_image",
                    "product_image": None,
                    "details": {}
                })
                
                # Send welcome message
                self.whatsapp_client.send_message(
                    from_number,
                    "Welcome to Marketing Image Editor! 📸\n\n"
                    "Please send your product image to begin.\n\n"
                    "After sending the image, I'll ask for details like company name, product name, price, etc."
                )
                
                log_and_print("INFO", f"Sent welcome message to {from_number}")
                return
            
            # Check for generate command
            if text.lower() == 'generate':
                print(f"[DEBUG] 'generate' command detected from {from_number}")
                log_and_print("INFO", f"User {user_id} sent 'generate' command")
                
                # Print current session state for debugging
                print(f"[DEBUG] Current session state: {session['state']}")
                print(f"[DEBUG] Product image path: {session.get('product_image')}")
                print(f"[DEBUG] Details collected: {json.dumps(session.get('details', {}))}")
                
                # Validate we have all required info
                if not session.get('product_image'):
                    log_and_print("WARNING", f"User {user_id} tried to generate without an image")
                    print(f"[DEBUG] No product image in session")
                    self.whatsapp_client.send_message(
                        from_number,
                        "Please send a product image first.\n"
                        "To start, send 'edit'."
                    )
                    return
                
                details = session.get('details', {})
                if not all([details.get('company_name'), details.get('product_name'), details.get('price')]):
                    missing = []
                    if not details.get('company_name'):
                        missing.append('company name')
                    if not details.get('product_name'):
                        missing.append('product name')
                    if not details.get('price'):
                        missing.append('price')
                    
                    log_and_print("WARNING", f"User {user_id} tried to generate with missing details: {missing}")
                    self.whatsapp_client.send_message(
                        from_number,
                        f"Missing required details: {', '.join(missing)}\n\n"
                        "Please provide all required information."
                    )
                    return
                
                # Process the request
                log_and_print("INFO", f"Starting image generation process for {user_id}")
                result = self.process_request(
                    user_id,
                    from_number,
                    session['product_image'],
                    details
                )
                
                if result['success']:
                    log_and_print("INFO", f"Image generated successfully for {user_id}: {result['image_path']}")
                    # Get local file path
                    image_path = result['image_path']
                    
                    # Verify the file exists
                    if not os.path.exists(image_path):
                        log_and_print("ERROR", f"Generated image file not found: {image_path}")
                        self.whatsapp_client.send_message(
                            from_number,
                            "Sorry, there was an error saving the generated image. Please try again by sending 'edit'."
                        )
                        return
                    
                    # Read the image for base64 encoding
                    log_and_print("INFO", f"Reading image file for base64 encoding: {image_path}")
                    print(f"[DEBUG] File size: {os.path.getsize(image_path)} bytes")
                    
                    try:
                        with open(image_path, 'rb') as img_file:
                            img_data = img_file.read()
                            print(f"[DEBUG] Image data read: {len(img_data)} bytes")
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            print(f"[DEBUG] Base64 encoding length: {len(img_base64)} characters")
                    except Exception as read_error:
                        log_and_print("ERROR", f"Failed to read image file: {str(read_error)}")
                        traceback.print_exc()
                        self.whatsapp_client.send_message(
                            from_number,
                            "Sorry, I had trouble processing the generated image. Please try again by sending 'edit'."
                        )
                        return
                    
                    # Send the generated image
                    log_and_print("INFO", f"Sending generated image to {from_number}")
                    print(f"[DEBUG] Sending image with base64 data and caption")
                    media_result = self.whatsapp_client.send_media(
                        to_number=from_number,
                        caption="🎉 Here's your marketing image!\n\n"
                                "To create another image, send 'edit' again.",
                        media_base64=img_base64,
                        filename=os.path.basename(image_path)
                    )
                    print(f"[DEBUG] Maytapi media send result: {json.dumps(media_result)}")
                    
                    if media_result['success']:
                        log_and_print("INFO", f"Image sent successfully to {from_number}")
                    else:
                        log_and_print("ERROR", f"Failed to send image: {media_result.get('error')}")
                        # Try to send an error message
                        self.whatsapp_client.send_message(
                            from_number,
                            "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                        )
                    
                    # End session
                    self.session_manager.end_session(from_number)
                    log_and_print("INFO", f"Session ended for user {user_id}")
                else:
                    log_and_print("ERROR", f"Failed to generate image: {result.get('error')}")
                    self.whatsapp_client.send_message(
                        from_number,
                        f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                        "Please try again by sending 'edit'."
                    )
                return
            
            # Handle conversation flow based on state
            if session['state'] == 'waiting_for_image':
                log_and_print("INFO", f"User {user_id} sent text while waiting for image")
                self.whatsapp_client.send_message(
                    from_number,
                    "Please send your product image first.\n"
                    "I'm waiting for an image file."
                )
                return
            
            elif session['state'] == 'waiting_for_details':
                log_and_print("INFO", f"User {user_id} sent details: {text}")
                # Parse the details
                lines = text.split('\n')
                detail_provided = False
                
                # First, parse details if user sent a structured message
                structured_details = {}
                
                # Check if this is a structured message
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if 'company' in key:
                            structured_details['company_name'] = value
                            detail_provided = True
                        elif 'product' in key:
                            structured_details['product_name'] = value
                            detail_provided = True
                        elif 'price' in key:
                            structured_details['price'] = value
                            detail_provided = True
                        elif 'tagline' in key:
                            structured_details['tagline'] = value
                            detail_provided = True
                        elif 'address' in key or 'location' in key:
                            structured_details['address'] = value
                            detail_provided = True
                
                # If structured details provided, update session
                if detail_provided:
                    # Get existing details and update with new ones
                    current_details = session.get('details', {})
                    current_details.update(structured_details)
                    
                    # Update session
                    self.session_manager.update_session(from_number, {
                        "details": current_details
                    })
                else:
                    # If no structured details, try to interpret single line input
                    current_details = session.get('details', {})
                    
                    if not current_details.get('company_name'):
                        current_details['company_name'] = text
                        log_and_print("INFO", f"Set company_name: {text}")
                        detail_provided = True
                    elif not current_details.get('product_name'):
                        current_details['product_name'] = text
                        log_and_print("INFO", f"Set product_name: {text}")
                        detail_provided = True
                    elif not current_details.get('price'):
                        current_details['price'] = text
                        log_and_print("INFO", f"Set price: {text}")
                        detail_provided = True
                    elif not current_details.get('tagline'):
                        current_details['tagline'] = text
                        log_and_print("INFO", f"Set tagline: {text}")
                        detail_provided = True
                    elif not current_details.get('address'):
                        current_details['address'] = text
                        log_and_print("INFO", f"Set address: {text}")
                        detail_provided = True
                    
                    # Update session with new details
                    self.session_manager.update_session(from_number, {
                        "details": current_details
                    })
                
                # Send updated status and next step
                log_and_print("INFO", f"Sending status update to {from_number}")
                session = self.session_manager.get_session(from_number)  # Get updated session
                details = session.get('details', {})
                
                status_msg = "📝 Current details:\n\n"
                status_msg += f"Company: {details.get('company_name', '❌')}\n"
                status_msg += f"Product: {details.get('product_name', '❌')}\n"
                status_msg += f"Price: {details.get('price', '❌')}\n"
                status_msg += f"Tagline: {details.get('tagline', '➖')}\n"
                status_msg += f"Address: {details.get('address', '➖')}\n\n"
                
                # Check what's still needed
                if not details.get('company_name'):
                    status_msg += "👉 Please send your company name.\n"
                elif not details.get('product_name'):
                    status_msg += "👉 Please send your product name.\n"
                elif not details.get('price'):
                    status_msg += "👉 Please send the price.\n"
                else:
                    status_msg += "✅ All required information received!\n\n"
                    status_msg += "To generate the marketing image, send 'generate'\n"
                    status_msg += "To add optional details (tagline, address), just send them."
                
                self.whatsapp_client.send_message(from_number, status_msg)
                return
            
            # Default state - waiting for command
            else:
                log_and_print("INFO", f"User {user_id} sent message in default state: '{text}'")
                
                # Check if we've already sent welcome message
                if not session.get('welcomed', False):
                    log_and_print("INFO", f"Sending initial welcome to user {user_id}")
                    self.whatsapp_client.send_message(
                        from_number,
                        "👋 Welcome to Marketing Image Generator!\n\n"
                        "To create a marketing image, send 'edit' to start."
                    )
                    # Mark this user as welcomed
                    self.session_manager.update_session(from_number, {
                        "welcomed": True
                    })
                else:
                    # For subsequent messages, just give a gentle reminder
                    log_and_print("INFO", f"User {user_id} sent unrecognized text: {text}")
                    self.whatsapp_client.send_message(
                        from_number,
                        "To start creating a marketing image, please send 'edit'."
                    )
                return
            
        except Exception as e:
            log_and_print("ERROR", f"Error handling text message: {str(e)}")
            traceback.print_exc()
            try:
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, an error occurred. Please try again.\n"
                    "Send 'edit' to start over."
                )
            except Exception as send_error:
                log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")
    
    def handle_image_message(self, from_number: str, media_data):
        """Handle incoming image messages
        
        Args:
            from_number: User's WhatsApp number
            media_data: Media data from WhatsApp
        """
        try:
            log_and_print("INFO", f"Image received from {from_number}. Processing...")
            print(f"[DEBUG] Media data type: {type(media_data)}")
            if isinstance(media_data, dict):
                print(f"[DEBUG] Media data keys: {list(media_data.keys())}")
            
            # Only process private chats (not groups) and only if from_number is valid
            if not from_number or "@g.us" in from_number:
                log_and_print("INFO", f"Message from group or invalid number: {from_number}. Ignored.")
                return
            
            # Get user's session
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            # Debug session state
            log_and_print("INFO", f"Current session state for user {user_id}: {session['state']}")
            
            # Check if we're in the right state to receive an image
            if session['state'] != 'waiting_for_image':
                log_and_print("WARNING", f"Received image but session state is {session['state']}, not waiting_for_image")
                self.whatsapp_client.send_message(
                    from_number,
                    "I wasn't expecting an image right now.\n"
                    "To start the process, please send 'edit' first."
                )
                return
            
            # Process the image from media_data
            try:
                # Extract image bytes
                image_bytes = self.image_handler.extract_image_from_whatsapp(media_data)
                
                # If we couldn't get image data, use a placeholder
                if not image_bytes:
                    log_and_print("WARNING", f"Could not extract image data from user {user_id} - creating placeholder")
                    image_bytes = self.image_handler.create_placeholder_image()
                    
                    # Let the user know about the limitation
                    self.whatsapp_client.send_message(
                        from_number,
                        "⚠️ I received your image but couldn't access its data.\n\n"
                        "I'll continue with a placeholder image for now. Your marketing image will still be generated based on your details."
                    )
                
                # Generate a unique filename and save the image
                filename = f"user_{user_id}_{int(datetime.now().timestamp())}.jpg"
                image_path, success = self.image_handler.save_image_from_bytes(image_bytes, user_id, filename)
                
                if not success or not image_path:
                    log_and_print("ERROR", f"Failed to save image for user {user_id}")
                    self.whatsapp_client.send_message(
                        from_number,
                        "Sorry, I couldn't process your image. Please try again by sending a different image."
                    )
                    return
                
                # Update session with image path and move to details state
                self.session_manager.update_session(from_number, {
                    "product_image": image_path,
                    "state": "waiting_for_details"
                })
                
                # Send confirmation and request details
                self.whatsapp_client.send_message(
                    from_number,
                    "✅ Product image received!\n\n"
                    "Now please provide the following details:\n\n"
                    "1️⃣ Company name\n"
                    "2️⃣ Product name\n"
                    "3️⃣ Price\n"
                    "4️⃣ Tagline (optional)\n"
                    "5️⃣ Address (optional)\n\n"
                    "You can send them one by one or all at once.\n"
                    "Example format:\n"
                    "Company: ABC Corp\n"
                    "Product: Premium Coffee\n"
                    "Price: $20\n\n"
                    "When you're ready to generate the image, send 'generate'"
                )
                
                log_and_print("INFO", f"Successfully processed image for user {user_id}")
                
            except Exception as process_error:
                log_and_print("ERROR", f"Error processing image: {str(process_error)}")
                traceback.print_exc()
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your image. Please try again.\n"
                    "Send 'edit' to start over."
                )
                
        except Exception as e:
            log_and_print("ERROR", f"Error handling image message: {str(e)}")
            traceback.print_exc()
            try:
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your image. Please try again.\n"
                    "Start over by sending 'edit'."
                )
            except Exception as send_error:
                log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")
    
    def handle_audio_message(self, from_number: str):
        """Handle incoming audio messages
        
        Args:
            from_number: User's WhatsApp number
        """
        try:
            log_and_print("INFO", f"Audio message received from {from_number}")
            
            # Get user's session
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            # Check if we've already welcomed the user
            if not session.get('welcomed', False):
                # Send welcome and mark as welcomed
                self.whatsapp_client.send_message(
                    from_number,
                    "👋 Welcome to Marketing Image Generator!\n\n"
                    "I received your voice message, but I work better with text.\n\n"
                    "To create a marketing image, send 'edit' to start."
                )
                self.session_manager.update_session(from_number, {
                    "welcomed": True
                })
            else:
                # Just send a helpful reminder
                self.whatsapp_client.send_message(
                    from_number,
                    "I received your voice message, but I work better with text.\n\n"
                    "To create a marketing image, please send 'edit'."
                )
            
            log_and_print("INFO", f"Audio message handled for user {user_id}")
            
        except Exception as e:
            log_and_print("ERROR", f"Error handling audio message: {str(e)}")
            traceback.print_exc()

###################
# SETUP AND INITIALIZATION
###################

# Initialize marketing bot
log_and_print("INFO", "Initializing marketing bot with environment variables")
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    maytapi_product_id=MAYTAPI_PRODUCT_ID,
    maytapi_api_token=MAYTAPI_API_TOKEN,
    maytapi_phone_id=MAYTAPI_PHONE_ID
)

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    log_and_print("INFO", "Home page accessed")
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
        "status": "running",
        "version": "2.0",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    log_and_print("DEBUG", "Health check accessed")
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "openai": OPENAI_API_KEY is not None,
        "maytapi": all([MAYTAPI_API_TOKEN, MAYTAPI_PRODUCT_ID, MAYTAPI_PHONE_ID]),
        "user_sessions": len(marketing_bot.session_manager.sessions)
    })

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    log_and_print("DEBUG", f"Serving image: {path}")
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via Maytapi webhook"""
    try:
        # Extract the webhook data
        webhook_data = request.json
        log_and_print("INFO", f"Received webhook data of type: {type(webhook_data)}")
        print(f"[DEBUG] Webhook data: {json.dumps(webhook_data)[:500]}...")
        
        # Check if we have a valid message structure
        if not webhook_data or not isinstance(webhook_data, dict):
            log_and_print("ERROR", "Invalid webhook data format")
            return jsonify({"status": "error", "message": "Invalid webhook data format"})
        
        # Extract message type and phone number
        message_type = webhook_data.get('type', '')
        if message_type != 'message':
            log_and_print("INFO", f"Ignoring non-message event: {message_type}")
            return jsonify({"status": "success", "message": "Non-message event ignored"})
        
        # Extract user and message data
        user_data = webhook_data.get('user', {})
        message_data = webhook_data.get('message', {})
        
        # Extract sender info
        from_number = user_data.get('id', '')  # Format: 905301234567@c.us
        phone_number = user_data.get('phone', '')  # Format: 905301234567
        
        # Use phone number if id is not available
        if not from_number and phone_number:
            from_number = f"{phone_number}@c.us"
        
        if not from_number:
            log_and_print("ERROR", "No sender phone number in webhook data")
            return jsonify({"status": "error", "message": "No sender phone number"})
        
        # Check if message is from the bot itself
        is_from_me = message_data.get('fromMe', False)
        if is_from_me:
            log_and_print("INFO", "Ignoring message from the bot itself")
            return jsonify({"status": "success", "message": "Bot message ignored"})
        
        # Extract message ID to prevent duplicate processing
        message_id = message_data.get('id', '')
        
        # Skip if we've already processed this message
        if message_id and message_id in processed_messages:
            log_and_print("INFO", f"Skipping duplicate message: {message_id}")
            return jsonify({"status": "success", "message": "Duplicate message skipped"})
        
        # Mark as processed if it has an ID
        if message_id:
            processed_messages[message_id] = datetime.now().timestamp()
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
        
        # Extract message content
        content_type = message_data.get('type', '')
        log_and_print("INFO", f"Message content type: {content_type}")
        
        # Handle media messages (images)
        if content_type == 'image':
            log_and_print("INFO", f"Detected image message from {from_number}")
            marketing_bot.handle_image_message(from_number, message_data)
            return jsonify({"status": "success", "message": "Image processed"})
        
        # Handle audio messages
        elif content_type in ['audio', 'voice', 'ptt']:
            log_and_print("INFO", f"Detected audio message from {from_number}")
            marketing_bot.handle_audio_message(from_number)
            return jsonify({"status": "success", "message": "Audio processed"})
        
        # Handle document messages that might be images
        elif content_type == 'document':
            log_and_print("INFO", f"Detected document message from {from_number}")
            # Check if it's an image document
            mime_type = message_data.get('mimetype', '')
            if mime_type and mime_type.startswith('image/'):
                log_and_print("INFO", f"Document is an image, processing as image")
                marketing_bot.handle_image_message(from_number, message_data)
            else:
                log_and_print("INFO", f"Document is not an image, sending reminder")
                marketing_bot.handle_audio_message(from_number)  # Reuse audio handler for generic response
            return jsonify({"status": "success", "message": "Document processed"})
        
        # Handle text messages
        elif content_type == 'text':
            body = message_data.get('text', '')
            log_and_print("INFO", f"Detected text message from {from_number}: {body}")
            marketing_bot.handle_text_message(from_number, body)
            return jsonify({"status": "success", "message": "Text processed"})
        
        # Fallback for other message types
        else:
            log_and_print("WARNING", f"Unhandled message type: {content_type}")
            # Send a generic response for unhandled types
            marketing_bot.handle_audio_message(from_number)  # Reuse audio handler for generic response
            return jsonify({"status": "success", "message": f"Unhandled message type: {content_type}"})
    
    except Exception as e:
        log_and_print("ERROR", f"Webhook error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

###################
# MAINTENANCE TASKS
###################

@app.route('/maintenance/cleanup', methods=['POST'])
def cleanup_sessions():
    """Cleanup old sessions and processed messages"""
    try:
        # Get cleanup hours from request or use default
        hours = request.json.get('hours', 24) if request.is_json else 24
        
        # Clean up old sessions
        session_count = marketing_bot.session_manager.clean_old_sessions(hours)
        
        # Clean up old processed messages
        current_time = datetime.now().timestamp()
        message_ids = list(processed_messages.keys())
        message_count = 0
        
        for msg_id in message_ids:
            if current_time - processed_messages[msg_id] > hours * 3600:
                processed_messages.pop(msg_id)
                message_count += 1
        
        log_and_print("INFO", f"Cleaned up {session_count} sessions and {message_count} processed messages older than {hours} hours")
        
        return jsonify({
            "status": "success",
            "cleaned_sessions": session_count,
            "cleaned_messages": message_count,
            "hours": hours
        })
    
    except Exception as e:
        log_and_print("ERROR", f"Maintenance cleanup error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Print environment info for debugging
    print("-" * 50)
    print("MARKETING BOT STARTUP")
    print("-" * 50)
    print(f"Python version: {os.sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables: PORT={port}")
    print(f"OpenAI API Key Present: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"Maytapi API Token Present: {'Yes' if MAYTAPI_API_TOKEN else 'No'}")
    print(f"Maytapi Product ID: {MAYTAPI_PRODUCT_ID}")
    print(f"Maytapi Phone ID: {MAYTAPI_PHONE_ID}")
    print(f"Checking directories...")
    
    # Make sure directories exist and are writable
    for directory in ['images', 'images/input', 'images/output']:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' exists: {os.path.exists(directory)}")
    
    # Test PIL import
    print(f"PIL/Pillow version: {Image.__version__ if hasattr(Image, '__version__') else 'unknown'}")
    
    # Log startup
    log_and_print("INFO", f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    print("-" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production