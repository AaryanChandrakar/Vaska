"""
Disease Prediction Chatbot - CLI Interface
Interactive chat interface for symptom-based disease prediction
"""

import sys
import argparse
from pathlib import Path

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

from symptom_extractor import SymptomExtractor
from predict import DiseaseInference
from conversation_manager import ConversationManager, ConversationState


class ChatbotCLI:
    """
    Command-line interface for disease prediction chatbot
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize chatbot
        
        Args:
            model_type: Type of model to use ('random_forest', 'xgboost', or 'mlp')
        """
        self.model_type = model_type
        
        # Initialize components
        print("Initializing Disease Prediction Chatbot...")
        print()
        
        try:
            self.extractor = SymptomExtractor()
            self.inference = DiseaseInference(
                model_path=f'models/{model_type}.pkl',
                model_type=model_type
            )
            self.conversation = ConversationManager()
            
            print(f"{self._color('✓ Chatbot ready!', 'green')}\n")
        except FileNotFoundError as e:
            print(f"{self._color('ERROR:', 'red')} {e}")
            print("\nPlease ensure you have:")
            print("  1. Run training: python train.py")
            print("  2. Generated processed_data/ and models/ directories")
            sys.exit(1)
        except Exception as e:
            print(f"{self._color('ERROR:', 'red')} Failed to initialize chatbot")
            print(f"Details: {e}")
            sys.exit(1)
    
    def _color(self, text: str, color: str) -> str:
        """
        Apply color to text if colorama is available
        
        Args:
            text: Text to color
            color: Color name
            
        Returns:
            Colored text
        """
        if not HAS_COLOR:
            return text
        
        colors = {
            'green': Fore.GREEN,
            'red': Fore.RED,
            'yellow': Fore.YELLOW,
            'blue': Fore.BLUE,
            'cyan': Fore.CYAN,
            'magenta': Fore.MAGENTA,
        }
        
        color_code = colors.get(color, '')
        return f"{color_code}{text}{Style.RESET_ALL}" if HAS_COLOR else text
    
    def _print_bot(self, message: str):
        """Print bot message"""
        print(f"\n{self._color('🤖 Bot:', 'cyan')} {message}")
    
    def _print_user(self, message: str):
        """Print user message"""
        print(f"\n{self._color('👤 You:', 'blue')} {message}")
    
    def _get_user_input(self) -> str:
        """Get user input"""
        try:
            prompt = f"\n{self._color('>', 'yellow')} "
            return input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return "exit"
    
    def _display_predictions(self, predictions: list, symptoms: list[str]):
        """
        Display predictions in a formatted way
        
        Args:
            predictions: List of (disease, confidence) tuples
            symptoms: List of symptoms used for prediction
        """
        print("\n" + "=" * 70)
        print(f"{self._color('🔍 DISEASE PREDICTION RESULTS', 'magenta')}")
        print("=" * 70)
        
        # Show symptoms
        print(f"\n{self._color('Symptoms analyzed:', 'cyan')}")
        for i, symptom in enumerate(symptoms, 1):
            symptom_display = symptom.replace('_', ' ').title()
            print(f"  {i}. {symptom_display}")
        
        # Show predictions
        print(f"\n{self._color('Possible conditions (ranked by confidence):', 'cyan')}\n")
        
        if not predictions:
            print("  No predictions available.")
        else:
            for i, (disease, confidence) in enumerate(predictions, 1):
                # Create confidence bar
                bar_length = int(confidence * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                
                # Color code by confidence
                if confidence >= 0.7:
                    disease_colored = self._color(disease, 'green')
                elif confidence >= 0.4:
                    disease_colored = self._color(disease, 'yellow')
                else:
                    disease_colored = disease
                
                print(f"  {i}. {disease_colored}")
                print(f"     {bar} {confidence:.1%}\n")
        
        # Medical disclaimer
        print("=" * 70)
        print(f"{self._color('⚕️  IMPORTANT:', 'red')}")
        print("These are computer-generated suggestions based on symptoms.")
        print("They are NOT a medical diagnosis.")
        print("Please consult a qualified healthcare professional for proper")
        print("examination, diagnosis, and treatment.")
        print("=" * 70)
    
    def run(self):
        """Run the chatbot main loop"""
        # Start conversation
        welcome = self.conversation.start_conversation()
        self._print_bot(welcome)
        
        while True:
            # Get user input
            user_input = self._get_user_input()
            
            if not user_input:
                continue
            
            # Extract symptoms
            try:
                extracted_symptoms = self.extractor.extract_symptoms_interactive(user_input)
            except Exception as e:
                print(f"{self._color('Warning:', 'yellow')} Symptom extraction error: {e}")
                extracted_symptoms = []
            
            # Process with conversation manager
            response = self.conversation.process_input(user_input, extracted_symptoms)
            
            # Handle action
            action = response.get('action')
            message = response.get('message', '')
            
            if action == 'exit':
                self._print_bot(message)
                break
            
            elif action == 'predict':
                symptoms = response.get('symptoms', [])
                
                if not symptoms:
                    self._print_bot("No symptoms collected. Please describe your symptoms first.")
                    continue
                
                # Make prediction
                try:
                    # Don't print intermediate output from inference
                    import io
                    import sys as _sys
                    old_stdout = _sys.stdout
                    _sys.stdout = io.StringIO()
                    
                    predictions = self.inference.predict_from_symptoms(
                        symptoms,
                        top_k=5,
                        use_enhanced_features=False
                    )
                    
                    _sys.stdout = old_stdout
                    
                    # Display results
                    self._display_predictions(predictions, symptoms)
                    
                    # Transition to follow-up
                    self.conversation.transition_to_follow_up()
                    
                    # Ask for next action
                    follow_up = "\nWould you like to:\n"
                    follow_up += "  • Start a new consultation? (type: new/restart)\n"
                    follow_up += "  • Exit? (type: exit/quit)"
                    self._print_bot(follow_up)
                    
                except Exception as e:
                    self._print_bot(f"Error making prediction: {e}")
                    print(f"Debug: {e}")
            
            elif message:
                self._print_bot(message)
        
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Disease Prediction Chatbot - Interactive symptom analyzer'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['random_forest', 'xgboost', 'mlp'],
        help='Model type to use (default: xgboost)'
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = ChatbotCLI(model_type=args.model)
        chatbot.run()
    except KeyboardInterrupt:
        print("\n\nChatbot interrupted. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
