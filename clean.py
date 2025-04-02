import pandas as pd
import re
import unicodedata
import sys
import os
from datetime import datetime

# Function to check if a character is in the Gujarati Unicode range
def is_gujarati(char):
    """Check if a character is in Gujarati Unicode range (0A80-0AFF)"""
    return '\u0A80' <= char <= '\u0AFF'

# Function to check if text contains actual Gujarati content
def has_gujarati_content(text):
    """Check if the text contains at least one Gujarati character"""
    if not isinstance(text, str):
        return False
    
    for char in text:
        if is_gujarati(char):
            return True
    return False

# Enhanced text cleaning function
def clean_text(text):
    if not isinstance(text, str):  
        return ""  # Handle non-string values
    
    # Remove emojis using Unicode blocks
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    text = emoji_pattern.sub('', text)
    
    # Remove all quote types
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace(''', '')
    text = text.replace(''', '')
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace('`', '')
    
    # Filter characters: keep only Gujarati, standard punctuation, and spaces
    cleaned_chars = []
    for char in text:
        # Keep Gujarati characters
        if is_gujarati(char):
            cleaned_chars.append(char)
        # Keep spaces
        elif char.isspace():
            cleaned_chars.append(char)
        # Keep limited standard punctuation
        elif char in '.,?!।॥':  # Including Gujarati purna viram and double danda
            cleaned_chars.append(char)
    
    cleaned_text = ''.join(cleaned_chars)
    
    # Remove extra spaces, including leading/trailing
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def main():
    # Set default paths or accept command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "sentences/gujarati_cricket_sentences_201_to_447.csv"
    
    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    output_path = f"cleaned_{base_name}.csv"
    
    print(f"📊 Processing: {input_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(input_path)
        
        # Check if "sentence" column exists
        if "sentence" not in df.columns:
            print("❌ Error: CSV file must contain a column named 'sentence'")
            return
        
        # Store original stats
        original_rows = len(df)
        original_chars = df["sentence"].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).sum()
        
        # Apply cleaning with progress indication
        print("🧹 Cleaning text...")
        df["clean_sentence"] = df["sentence"].apply(clean_text)
        
        # Count rows before filtering
        rows_after_cleaning = len(df)
        
        # Drop empty rows or rows with only punctuation and whitespace
        df = df[df["clean_sentence"].apply(has_gujarati_content)]
        
        # Calculate stats
        cleaned_rows = len(df)
        dropped_rows = original_rows - cleaned_rows
        punctuation_only_rows = rows_after_cleaning - cleaned_rows
        cleaned_chars = df["clean_sentence"].apply(len).sum()
        chars_removed = original_chars - cleaned_chars
        
        # Save cleaned dataset
        print(f"💾 Saving cleaned dataset to: {output_path}")
        df[["clean_sentence"]].to_csv(output_path, index=False)
        
        # Display statistics
        print("\n📈 Cleaning Statistics:")
        print(f"   - Original rows: {original_rows}")
        print(f"   - Cleaned rows: {cleaned_rows}")
        print(f"   - Rows dropped: {dropped_rows} ({(dropped_rows/original_rows*100):.2f}% of total)")
        print(f"   - Rows with only punctuation removed: {punctuation_only_rows}")
        print(f"   - Original characters: {original_chars}")
        print(f"   - Cleaned characters: {cleaned_chars}")
        print(f"   - Characters removed: {chars_removed} ({(chars_removed/original_chars*100):.2f}% of total)")
        
        # Display a few examples
        print("\n📝 Sample of cleaned sentences:")
        sample_size = min(5, len(df))
        for i in range(sample_size):
            original = str(df.iloc[i]["sentence"])
            cleaned = df.iloc[i]["clean_sentence"]
            if len(original) > 70:
                original = original[:67] + "..."
            if len(cleaned) > 70:
                cleaned = cleaned[:67] + "..."
            print(f"   Original: {original}")
            print(f"   Cleaned:  {cleaned}")
            print()
        
        print(f"✅ Cleaning complete! File saved as: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found.")
    except pd.errors.EmptyDataError:
        print("❌ Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("❌ Error: Unable to parse the CSV file. Check if it's a valid CSV format.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()