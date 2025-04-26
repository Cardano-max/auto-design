#!/usr/bin/env python3
"""
Marketing Image Generator
------------------------
Automates the creation of marketing images using ChatGPT.
Takes product images and information, processes them through ChatGPT,
and generates professional marketing materials.
"""

import os
import time
import base64
import json
import logging
import argparse
import urllib.request
import platform
import subprocess
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marketing_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBot")

# Create Flask app for the demo interface
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store user sessions
user_sessions = {}

# CSS Selectors for ChatGPT
CHATGPT_SELECTORS = {
    'text_input': '#prompt-textarea',
    'submit_button': '#composer-submit-button',
    'output_text': '.markdown.prose.dark\\:prose-invert.w-full.break-words.dark',
    'generated_image': 'img.absolute.top-0.z-1.w-full',
    'image_upload_button': 'button.h-full.w-full'
}

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details):
        """Generate the master prompt template with product details"""
        return f"""Transform this product photo into a professional marketing poster with the following specifications:

1. PRODUCT PRESENTATION:
   - Place the product as the central focal point
   - Remove the original background completely
   - Apply subtle product shadows for dimension

2. BRANDING ELEMENTS:
   - Add company name "{product_details.get('company_name', '')}" at the top in an elegant, readable font
   - {f'Include company logo in the top left corner' if product_details.get('logo') else ''}
   - Add product name "{product_details.get('product_name', '')}" below the product image in bold, attractive typography
   - Create a circular price badge showing "{product_details.get('price', '')}" in the bottom right corner using a standout color (maroon/red) with white text

3. MARKETING ELEMENTS:
   - Add tagline "{product_details.get('tagline', '')}" in italic font between the company name and product image
   - Include location/address "{product_details.get('address', '')}" in smaller font below the product name
   - Use a clean, appealing background in {product_details.get('background_style', 'light beige')} with subtle texture/gradient

4. DESIGN STYLE:
   - Use professional food/beverage marketing aesthetics similar to high-end café materials
   - Maintain balanced spacing between all elements
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable and properly positioned

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details):
        """Generate beverage-specific prompt template"""
        return f"""Transform this beverage photo into a premium café marketing poster:

1. PRODUCT ENHANCEMENT:
   - Remove the original background completely
   - Keep any whipped cream, garnishes, and special toppings visible and appetizing
   - Add delicate steam effect for hot drinks or condensation for cold drinks
   - Enhance the color of the beverage to look more vibrant and appealing

2. BRANDING PLACEMENT:
   - Position "{product_details.get('company_name', '')}" at top in an elegant script or modern sans-serif font
   - Place product name "{product_details.get('product_name', '')}" below the product
   - Create a star-shaped or circular price tag showing "{product_details.get('price', '')}" in the bottom corner
   - Add "{product_details.get('tagline', '')}" below the company name

3. AESTHETIC ELEMENTS:
   - Use a warm, inviting background color that complements the drink color
   - Add subtle coffee beans, cocoa nibs, or relevant ingredients floating in the background
   - Apply a soft vignette effect to create focus on the drink
   - Include location "{product_details.get('address', '')}" in a smaller, elegant font at the bottom

The final poster should evoke premium café branding like Starbucks or specialty coffee shops, with perfectly balanced design elements and professional marketing appeal.
"""

    @staticmethod
    def get_food_template(product_details):
        """Generate food-specific prompt template"""
        return f"""Transform this food product photo into an appetizing marketing poster:

1. PRODUCT PRESENTATION:
   - Remove the original background completely
   - Enhance food details, textures, and colors for maximum appetite appeal
   - Add subtle steam effects for hot items or fresh/dewy appearance for cold items
   - Ensure the food looks perfectly prepared and delicious

2. BRANDING ELEMENTS:
   - Place "{product_details.get('company_name', '')}" at the top in a clean, food-appropriate font
   - Position product name "{product_details.get('product_name', '')}" below the food in an appealing, bold font
   - Create an eye-catching price tag showing "{product_details.get('price', '')}" in a complementary color
   - Add a mouth-watering tagline "{product_details.get('tagline', '')}" that emphasizes flavor or freshness

3. MARKETING ENHANCEMENTS:
   - Use a background color/pattern that complements the food's colors
   - Add subtle ingredients or garnishes in the background for context
   - Include location/address "{product_details.get('address', '')}" at the bottom in a clean, readable font
   - Balance all elements for a professional food marketing poster

The final design should resemble professional food photography marketing used by premium restaurants or bakeries, with perfect spacing and appetizing presentation.
"""

    @staticmethod
    def get_generation_retry_prompt():
        """Prompt to retry image generation if no image is found in response"""
        return """Based on my description, please generate a marketing image for this product. 
Make sure to remove the background, add the branding elements (company name, product name, price) 
and create the professional marketing poster as described. Generate the image now."""

###################
# UTILITY FUNCTIONS
###################

def get_chromedriver_path():
    """Get the ChromeDriver path based on the operating system"""
    try:
        # Try to find ChromeDriver in PATH
        if platform.system() == "Windows":
            chromedriver_name = "chromedriver.exe"
        else:
            chromedriver_name = "chromedriver"
            
        # Check if ChromeDriver is in PATH
        path_result = subprocess.run(['which', chromedriver_name] if platform.system() != "Windows" 
                                    else ['where', chromedriver_name], 
                                    capture_output=True, text=True)
        
        if path_result.returncode == 0 and path_result.stdout.strip():
            logger.info(f"Found ChromeDriver in PATH: {path_result.stdout.strip()}")
            return path_result.stdout.strip()
        
        # For Mac, try typical homebrew locations
        if platform.system() == "Darwin":  # macOS
            homebrew_paths = [
                "/usr/local/bin/chromedriver",
                "/opt/homebrew/bin/chromedriver",
                os.path.expanduser("~/.webdriverManager/chromedriver/mac64/chromedriver")
            ]
            
            for path in homebrew_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    logger.info(f"Found ChromeDriver at: {path}")
                    return path
                    
        # Look in current directory or subdirectories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(current_dir):
            if chromedriver_name in files:
                driver_path = os.path.join(root, chromedriver_name)
                # Make it executable on Unix-like systems
                if platform.system() != "Windows":
                    os.chmod(driver_path, 0o755)
                logger.info(f"Found ChromeDriver in project directory: {driver_path}")
                return driver_path
        
        logger.warning("ChromeDriver not found automatically. Please provide path with --chromedriver option.")
        return None
    except Exception as e:
        logger.error(f"Error finding ChromeDriver: {str(e)}")
        return None


def save_base64_image(base64_string, output_path):
    """Save a base64 image to a file"""
    try:
        # Remove data URL prefix if it exists
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode and save the image
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(base64_string))
        
        return True
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}")
        return False


def save_image_from_url(url, output_path):
    """Download and save an image from a URL"""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        return False


def generate_timestamp_filename(prefix="image", extension="png"):
    """Generate a unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


###################
# CHATGPT AUTOMATION
###################

class ChatGPTAutomation:
    def __init__(self, headless=True, chrome_driver_path=None):
        """Initialize the ChatGPT automation class"""
        self.headless = headless
        self.driver = None
        self.wait = None
        self.logged_in = False
        self.chrome_driver_path = chrome_driver_path
    
    def start_browser(self):
        """Start a new browser session"""
        logger.info("Starting browser session")
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-notifications")
        options.add_argument("--window-size=1920,1080")
        
        try:
            # First try with provided chrome_driver_path
            if self.chrome_driver_path and os.path.exists(self.chrome_driver_path):
                logger.info(f"Using provided ChromeDriver at: {self.chrome_driver_path}")
                service = Service(executable_path=self.chrome_driver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                # Try to find ChromeDriver in system
                detected_path = get_chromedriver_path()
                if detected_path:
                    logger.info(f"Using detected ChromeDriver at: {detected_path}")
                    service = Service(executable_path=detected_path)
                    self.driver = webdriver.Chrome(service=service, options=options)
                else:
                    # Fall back to letting Selenium find ChromeDriver
                    logger.info("No ChromeDriver path provided, letting Selenium auto-detect")
                    self.driver = webdriver.Chrome(options=options)
                    
            self.wait = WebDriverWait(self.driver, 20)
            return self.driver
        except Exception as e:
            logger.error(f"Error starting browser: {str(e)}")
            raise
    
    def login_to_chatgpt(self, username=None, password=None):
        """Login to ChatGPT (this is a placeholder - you'll need to implement actual login)"""
        logger.info("Logging into ChatGPT")
        
        # For demo purposes, we're just navigating to ChatGPT
        # In a real implementation, you'd handle the login flow
        try:
            self.driver.get("https://chat.openai.com/")
            time.sleep(3)
            
            # Check if already logged in by looking for the text input
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input'])))
                logger.info("Already logged in to ChatGPT")
                self.logged_in = True
                return True
            except TimeoutException:
                logger.info("Not logged in yet, looking for login elements")
            
            # Try to handle login page
            try:
                # Click "Log in" button if found
                login_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Log in')]")))
                login_button.click()
                time.sleep(2)
                
                # Enter email and password if provided
                if username and password:
                    email_input = self.wait.until(EC.presence_of_element_located((By.ID, "username")))
                    email_input.send_keys(username)
                    time.sleep(1)
                    
                    continue_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]")))
                    continue_button.click()
                    time.sleep(2)
                    
                    password_input = self.wait.until(EC.presence_of_element_located((By.ID, "password")))
                    password_input.send_keys(password)
                    time.sleep(1)
                    
                    submit_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
                    submit_button.click()
                    time.sleep(5)
                    
                    # Check if login was successful
                    try:
                        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input'])))
                        logger.info("Login successful")
                        self.logged_in = True
                        return True
                    except TimeoutException:
                        logger.error("Login failed - could not find text input element")
                        return False
                else:
                    logger.warning("No username/password provided - please log in manually within 60 seconds")
                    # Give user 60 seconds to log in manually
                    try:
                        self.wait = WebDriverWait(self.driver, 60)  # Increase wait time for manual login
                        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input'])))
                        logger.info("Manual login successful")
                        self.logged_in = True
                        self.wait = WebDriverWait(self.driver, 20)  # Reset wait time
                        return True
                    except TimeoutException:
                        logger.error("Manual login timed out")
                        return False
            except Exception as login_error:
                logger.warning(f"Error during login flow: {str(login_error)}")
                logger.warning("Attempting to handle manual login...")
                
                # Give user 60 seconds to log in manually
                try:
                    self.wait = WebDriverWait(self.driver, 60)  # Increase wait time for manual login
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input'])))
                    logger.info("Manual login successful")
                    self.logged_in = True
                    self.wait = WebDriverWait(self.driver, 20)  # Reset wait time
                    return True
                except TimeoutException:
                    logger.error("Manual login timed out")
                    return False
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False
    
    def start_new_chat(self):
        """Start a new chat in ChatGPT"""
        logger.info("Starting new chat")
        try:
            # Try to find and click the "New chat" button
            try:
                new_chat_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'New chat')]")))
                new_chat_button.click()
                time.sleep(2)
                return True
            except (TimeoutException, NoSuchElementException):
                # Try alternative selector
                logger.info("Trying alternative selector for New chat button")
                new_chat_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "nav a.rounded-md")))
                new_chat_button.click()
                time.sleep(2)
                return True
        except Exception as e:
            logger.error(f"Error starting new chat: {str(e)}")
            
            # If we couldn't start a new chat, try refreshing the page
            try:
                logger.info("Refreshing page to start fresh chat")
                self.driver.refresh()
                time.sleep(3)
                return True
            except Exception as refresh_error:
                logger.error(f"Error refreshing page: {str(refresh_error)}")
                return False
    
    def send_message(self, message):
        """Send a message to ChatGPT"""
        logger.info(f"Sending message to ChatGPT: {message[:50]}...")
        
        try:
            # Wait for the text input to be available
            text_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input'])))
            
            # Clear any existing text
            text_input.clear()
            
            # Type the message
            # For longer messages, we'll type in chunks to avoid issues
            chunk_size = 100
            for i in range(0, len(message), chunk_size):
                chunk = message[i:i+chunk_size]
                text_input.send_keys(chunk)
                time.sleep(0.5)
            
            time.sleep(1)
            
            # Click the submit button
            submit_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, CHATGPT_SELECTORS['submit_button'])))
            submit_button.click()
            
            # Wait for the response to complete
            self._wait_for_response_complete()
            
            # Get the response text
            response = self._get_last_response()
            
            return response
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return None
    
    def upload_image(self, image_path):
        """Upload an image to ChatGPT"""
        logger.info(f"Uploading image: {image_path}")
        
        try:
            # Find and click the image upload button
            try:
                upload_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, CHATGPT_SELECTORS['image_upload_button'])))
                upload_button.click()
                time.sleep(1)
            except (TimeoutException, NoSuchElementException):
                # Try alternative approach - looking for the paperclip button
                logger.info("Trying alternative selector for upload button")
                upload_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Attach files']")))
                upload_button.click()
                time.sleep(1)
                
            # Find the file input and upload the image
            try:
                file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
                file_input.send_keys(os.path.abspath(image_path))
            except NoSuchElementException:
                # Try JavaScript approach to trigger file input
                logger.info("Using JavaScript to trigger file input")
                file_input = self.driver.execute_script("""
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.style.display = 'none';
                    document.body.appendChild(input);
                    return input;
                """)
                file_input.send_keys(os.path.abspath(image_path))
            
            # Wait for image to upload
            time.sleep(5)
            
            logger.info("Image uploaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            traceback.print_exc()
            return False
    
    def _wait_for_response_complete(self):
        """Wait for ChatGPT's response to complete"""
        logger.info("Waiting for ChatGPT response...")
        
        max_wait_time = 180  # 3 minutes maximum wait
        start_time = time.time()
        
        try:
            # First check if we can see the "Stop generating" button
            try:
                WebDriverWait(self.driver, 2).until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Stop generating')]")))
                logger.info("ChatGPT is generating a response...")
                
                # Wait for the button to disappear (response complete)
                WebDriverWait(self.driver, max_wait_time).until_not(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Stop generating')]"))
                )
            except TimeoutException:
                # If we never saw the stop button, that's fine
                pass
            
            # Additional check: wait until the text input is enabled again
            end_time = time.time() + max_wait_time - (time.time() - start_time)
            try:
                WebDriverWait(self.driver, end_time - time.time()).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, CHATGPT_SELECTORS['text_input']))
                )
            except TimeoutException:
                logger.warning("Text input not clickable after waiting for response, but proceeding anyway")
        except Exception as e:
            logger.warning(f"Warning while waiting for response: {str(e)}")
        
        # Additional wait to ensure response is fully loaded
        time.sleep(3)
        logger.info("ChatGPT response complete")
    
    def _get_last_response(self):
        """Get the last response from ChatGPT"""
        try:
            # Find all response elements
            retries = 3
            while retries > 0:
                try:
                    response_elements = self.driver.find_elements(By.CSS_SELECTOR, CHATGPT_SELECTORS['output_text'])
                    break
                except StaleElementReferenceException:
                    retries -= 1
                    time.sleep(1)
            
            if not response_elements:
                logger.warning("No response elements found")
                return None
            
            # Get the text from the last response
            last_response = response_elements[-1].text
            logger.info(f"Got response: {last_response[:50]}...")
            return last_response
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return None
    
    def check_for_generated_image(self, max_retries=3):
        """Check if ChatGPT has generated an image and download it"""
        logger.info("Checking for generated image...")
        
        for attempt in range(max_retries):
            try:
                # Wait for image to appear
                try:
                    image_element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CHATGPT_SELECTORS['generated_image'])))
                    
                    # Get image source
                    image_src = image_element.get_attribute('src')
                    
                    if image_src and image_src.startswith('data:image'):
                        # Image is a data URL
                        logger.info("Found image as data URL")
                        return image_src
                    elif image_src:
                        # Image is a regular URL
                        logger.info(f"Found image at URL: {image_src}")
                        return image_src
                    else:
                        logger.warning("Image element found but no src attribute")
                except TimeoutException:
                    logger.warning(f"No image found on attempt {attempt+1}/{max_retries}")
                    
                    # Try alternative selectors if first attempt failed
                    try:
                        # Try looking for any img element in the latest response
                        latest_response = self.driver.find_elements(By.CSS_SELECTOR, CHATGPT_SELECTORS['output_text'])[-1]
                        image_element = latest_response.find_element(By.TAG_NAME, "img")
                        
                        # Get image source
                        image_src = image_element.get_attribute('src')
                        if image_src:
                            logger.info(f"Found image with alternative selector: {image_src[:50]}")
                            return image_src
                    except (IndexError, NoSuchElementException):
                        logger.warning("No image found with alternative selector")
                
                if attempt < max_retries - 1:
                    time.sleep(3)  # Wait between retries
            except Exception as e:
                logger.error(f"Error checking for image on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Wait between retries
        
        logger.error("No image found after all retry attempts")
        return None
    
    def generate_marketing_image(self, image_path, product_details, product_type="beverage"):
        """Generate a marketing image based on the product details"""
        logger.info(f"Generating marketing image for {product_details.get('product_name', 'product')}")
        
        if not self.logged_in:
            logger.error("Not logged in to ChatGPT")
            return None
        
        try:
            # Upload the product image
            upload_success = self.upload_image(image_path)
            if not upload_success:
                logger.error("Failed to upload image")
                return None
            
            # Select the appropriate prompt template based on product type
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
            else:
                prompt = PromptTemplates.get_master_template(product_details)
            
            # Send the prompt
            self.send_message(prompt)
            
            # Wait for the response and check for generated image
            time.sleep(5)  # Give ChatGPT a moment to process
            image_src = self.check_for_generated_image()
            
            # If no image found, try asking explicitly for an image
            retry_count = 0
            max_retries = 2
            
            while not image_src and retry_count < max_retries:
                retry_count += 1
                logger.warning(f"No image found, trying explicit request (attempt {retry_count}/{max_retries})")
                
                retry_prompt = PromptTemplates.get_generation_retry_prompt()
                self.send_message(retry_prompt)
                
                # Give more time for the retry
                time.sleep(8)
                image_src = self.check_for_generated_image()
            
            if image_src:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                if image_src.startswith('data:image'):
                    # Save data URL as image
                    success = save_base64_image(image_src, output_path)
                else:
                    # Download from URL
                    success = save_image_from_url(image_src, output_path)
                
                if success:
                    logger.info(f"Saved generated image to {output_path}")
                    return output_path
                else:
                    logger.error("Failed to save image")
                    return None
            else:
                logger.error("No image generated after retries")
                return None
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def close(self):
        """Close the browser"""
        if self.driver:
            logger.info("Closing browser")
            self.driver.quit()


###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, chrome_driver_path=None):
        """Initialize the marketing bot"""
        self.chatgpt_bot = None
        self.chrome_driver_path = chrome_driver_path
    
    def initialize_chatgpt(self, headless=True):
        """Initialize the ChatGPT automation"""
        if self.chatgpt_bot:
            self.chatgpt_bot.close()
        
        try:
            self.chatgpt_bot = ChatGPTAutomation(headless=headless, chrome_driver_path=self.chrome_driver_path)
            self.chatgpt_bot.start_browser()
            login_success = self.chatgpt_bot.login_to_chatgpt()
            
            if login_success:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error initializing ChatGPT: {str(e)}")
            return False
    
    def process_request(self, user_id, product_image_path, product_details, product_type="beverage"):
        """Process a marketing image request"""
        logger.info(f"Processing request for user {user_id}")
        
        # Initialize ChatGPT if needed
        if not self.chatgpt_bot or not self.chatgpt_bot.logged_in:
            if not self.initialize_chatgpt():
                return {
                    "success": False,
                    "error": "Failed to initialize ChatGPT"
                }
        
        # Start new chat
        self.chatgpt_bot.start_new_chat()
        
        # Generate marketing image
        output_path = self.chatgpt_bot.generate_marketing_image(
            product_image_path, 
            product_details,
            product_type
        )
        
        if output_path:
            return {
                "success": True,
                "image_path": output_path,
                "image_url": f"/images/output/{os.path.basename(output_path)}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate marketing image"
            }
    
    def close(self):
        """Close the marketing bot"""
        if self.chatgpt_bot:
            self.chatgpt_bot.close()


###################
# FLASK WEB APP
###################

# Initialize marketing bot
chrome_driver_path = None  # Set this to your ChromeDriver path if needed
marketing_bot = MarketingBot(chrome_driver_path=chrome_driver_path)

# Flask routes for the demo
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"})
        
        # Generate a unique filename
        filename = f"product_{int(time.time())}.png"
        filepath = os.path.join('images/input', filename)
        file.save(filepath)
        
        # Create or update user session
        user_id = request.form.get('user_id', f"user_{int(time.time())}")
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                "product_image": filepath,
                "details": {}
            }
        else:
            user_sessions[user_id]["product_image"] = filepath
        
        return jsonify({
            "success": True,
            "message": "Image uploaded successfully",
            "user_id": user_id,
            "image_url": f"/images/input/{filename}"
        })
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/update-details', methods=['POST'])
def update_details():
    """Update product details"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id or user_id not in user_sessions:
            return jsonify({"success": False, "error": "Invalid user ID"})
        
        # Update details
        user_sessions[user_id]["details"] = {
            "company_name": data.get('company_name', ''),
            "product_name": data.get('product_name', ''),
            "price": data.get('price', ''),
            "tagline": data.get('tagline', ''),
            "address": data.get('address', ''),
            "background_style": data.get('background_style', 'light beige')
        }
        
        return jsonify({
            "success": True,
            "message": "Details updated successfully"
        })
    except Exception as e:
        logger.error(f"Error updating details: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """Generate marketing image"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id or user_id not in user_sessions:
            return jsonify({"success": False, "error": "Invalid user ID"})
        
        # Get product details
        product_image = user_sessions[user_id]["product_image"]
        product_details = user_sessions[user_id]["details"]
        product_type = data.get('product_type', 'beverage')
        
        # Check if we have the minimum required details
        if not product_image:
            return jsonify({"success": False, "error": "Product image is required"})
        
        if not product_details.get('company_name'):
            return jsonify({"success": False, "error": "Company name is required"})
        
        if not product_details.get('product_name'):
            return jsonify({"success": False, "error": "Product name is required"})
        
        if not product_details.get('price'):
            return jsonify({"success": False, "error": "Price is required"})
        
        # Process the request
        result = marketing_bot.process_request(
            user_id,
            product_image,
            product_details,
            product_type
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/initialize-bot', methods=['POST'])
def initialize_bot():
    """Initialize the marketing bot"""
    try:
        headless = request.json.get('headless', True)
        success = marketing_bot.initialize_chatgpt(headless=headless)
        
        return jsonify({
            "success": success,
            "message": "Bot initialized successfully" if success else "Failed to initialize bot"
        })
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Marketing Bot')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--chromedriver', type=str, help='Path to ChromeDriver executable')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    args = parser.parse_args()
    
    # Set ChromeDriver path if provided
    if args.chromedriver:
        chrome_driver_path = args.chromedriver
        marketing_bot = MarketingBot(chrome_driver_path=chrome_driver_path)
    
    # Initialize the marketing bot
    try:
        marketing_bot.initialize_chatgpt(headless=args.headless)
    except Exception as e:
        logger.error(f"Error initializing marketing bot: {str(e)}")
    
    # Run the Flask app
    app.run(debug=True, port=args.port)