"""
Translation utilities for English-Bengali bidirectional translation
Uses BanglaT5 for translation tasks
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

class BanglaT5Translator:
    """
    Handles English ↔ Bengali translation using BanglaT5
    """
    
    def __init__(self):
        self.model_bn_en = None  # Bengali to English
        self.tokenizer_bn_en = None
        self.model_en_bn = None  # English to Bengali
        self.tokenizer_en_bn = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized_bn_en = False
        self._initialized_en_bn = False
        
    def _lazy_load_bn_en(self):
        """Load Bengali→English model only when needed"""
        if self._initialized_bn_en:
            return
            
        print("🔄 Loading BanglaT5 BN→EN translation model...")
        try:
            # Bengali to English model
            model_dir = r"D:\Edtech\model_cache\models--csebuetnlp--banglat5_nmt_bn_en\snapshots\997417a326d498848f501f5b9a7c6995684b2402"
            
            print(f"📂 Loading BN→EN from: {model_dir}")
            self.tokenizer_bn_en = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model_bn_en = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
            self.model_bn_en.to(self.device)
            self.model_bn_en.eval()
            self._initialized_bn_en = True
            print(f"✅ BanglaT5 BN→EN loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load BanglaT5 BN→EN: {e}")
            print("⚠️ Make sure the model exists at the specified path")
            raise
    
    def _lazy_load_en_bn(self):
        """Load English→Bengali model only when needed"""
        if self._initialized_en_bn:
            return
            
        print("🔄 Loading BanglaT5 EN→BN translation model...")
        try:
            # English to Bengali model
            model_dir = r"D:\Edtech\model_cache\banglat5_nmt_en_bn"
            
            print(f"📂 Loading EN→BN from: {model_dir}")
            self.tokenizer_en_bn = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model_en_bn = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
            self.model_en_bn.to(self.device)
            self.model_en_bn.eval()
            self._initialized_en_bn = True
            print(f"✅ BanglaT5 EN→BN loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load BanglaT5 EN→BN: {e}")
            print("⚠️ Falling back to BN→EN model for reverse translation (may be less accurate)")
            # Fallback: use the bn_en model in reverse
            self._lazy_load_bn_en()
            self._initialized_en_bn = True
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is Bengali or English
        Returns: 'bn' or 'en'
        """
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'en'
        
        bengali_ratio = bengali_chars / total_chars
        return 'bn' if bengali_ratio > 0.3 else 'en'
    
    def translate_en_to_bn(self, text: str) -> str:
        """Translate English to Bengali"""
        self._lazy_load_en_bn()
        
        try:
            print(f"🔄 Translating EN→BN: {text[:100]}...")
            
            # Check if we have the proper EN→BN model
            if self.model_en_bn and self.tokenizer_en_bn:
                # Use proper EN→BN model
                input_text = f"translate English to Bengali: {text}"
                inputs = self.tokenizer_en_bn(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model_en_bn.generate(
                        **inputs,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
                
                translated = self.tokenizer_en_bn.decode(outputs[0], skip_special_tokens=True)
            else:
                # Fallback: Use BN→EN model with modified prompt (less accurate)
                print("⚠️ Using BN→EN model in reverse mode")
                input_text = f"translate English to Bengali: {text}"
                inputs = self.tokenizer_bn_en(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model_bn_en.generate(
                        **inputs,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
                
                translated = self.tokenizer_bn_en.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the translation - remove common prefixes
            translated = translated.strip()
            
            # Remove Bengali prefixes
            prefixes_to_remove = [
                'ইংরেজি থেকে বাংলা অনুবাদ:',
                'ইংরেজি থেকে বাংলা অনুবাদ :',
                'অনুবাদ:',
                'অনুবাদ :',
                'বাংলা:',
                'বাংলা :',
                'English to Bengali translation:',
                'Translation:',
                'Bengali:'
            ]
            
            for prefix in prefixes_to_remove:
                if translated.startswith(prefix):
                    translated = translated[len(prefix):].strip()
                    break
            
            print(f"✅ Translation result: {translated[:100]}...")
            return translated
            
        except Exception as e:
            print(f"⚠️ Translation EN→BN failed: {e}")
            return text  # Return original if translation fails
    
    def translate_bn_to_en(self, text: str) -> str:
        """Translate Bengali to English"""
        self._lazy_load_bn_en()
        
        try:
            print(f"🔄 Translating BN→EN: {text[:100]}...")
            
            # Prepare input with task prefix
            input_text = f"translate Bengali to English: {text}"
            inputs = self.tokenizer_bn_en(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model_bn_en.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode
            translated = self.tokenizer_bn_en.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the translation - remove common prefixes
            translated = translated.strip()
            
            # Remove English prefixes
            prefixes_to_remove = [
                'Translate Bengali to English:',
                'Bengali to English translation:',
                'Translation:',
                'English:',
                'বাংলা থেকে ইংরেজি অনুবাদ:',
                'অনুবাদ:'
            ]
            
            for prefix in prefixes_to_remove:
                if translated.lower().startswith(prefix.lower()):
                    translated = translated[len(prefix):].strip()
                    break
            
            print(f"✅ Translation result: {translated[:100]}...")
            return translated
            
        except Exception as e:
            print(f"⚠️ Translation BN→EN failed: {e}")
            return text  # Return original if translation fails
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Generic translation method
        
        Args:
            text: Text to translate
            source_lang: 'en' or 'bn'
            target_lang: 'en' or 'bn'
        
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text  # No translation needed
        
        if source_lang == 'en' and target_lang == 'bn':
            return self.translate_en_to_bn(text)
        elif source_lang == 'bn' and target_lang == 'en':
            return self.translate_bn_to_en(text)
        else:
            print(f"⚠️ Unsupported language pair: {source_lang} → {target_lang}")
            return text


# Singleton instance
_translator_instance: Optional[BanglaT5Translator] = None

def get_translator() -> BanglaT5Translator:
    """Get or create singleton translator instance"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = BanglaT5Translator()
    return _translator_instance