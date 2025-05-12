# ============================================
# Step 1: Install Required Libraries
# ============================================
!pip install transformers sentencepiece nltk --quiet

# ============================================
# Step 2: Import Libraries
# ============================================
import re
import nltk
from transformers import pipeline

nltk.download('punkt')

# ============================================
# Step 3: Document Segmentation
# ============================================
def segment_document(text):
    pattern = re.compile(r'(Clause\s*\d+\.?\d*|Section\s*\d+\.?\d*)', re.IGNORECASE)
    parts = pattern.split(text)
    headers = pattern.findall(text)
    clauses = [f"{hdr.strip()} {part.strip()}" for hdr, part in zip(headers, parts[1:])]
    return clauses

# ============================================
# Step 4: Key Clause Identification
# ============================================
def identify_key_clauses(clauses, keywords=None):
    if keywords is None:
        keywords = ["termination", "liability", "confidentiality", "indemnity", "jurisdiction", "payment"]
    key_clauses = [clause for clause in clauses if any(kw in clause.lower() for kw in keywords)]
    return key_clauses

# ============================================
# Step 5: Load Summarization Model
# ============================================
def load_simplifier_model():
    return pipeline("summarization", model="google/flan-t5-base")

# ============================================
# Step 6: Clause Simplification
# ============================================
def simplify_clauses(clauses, simplifier):
    simplified = []
    for clause in clauses:
        if len(clause.split()) > 5:
            try:
                result = simplifier(clause, max_length=100, min_length=30, do_sample=False)
                simplified.append(result[0]['summary_text'])
            except Exception as e:
                simplified.append(f"Error simplifying clause: {e}")
        else:
            simplified.append(clause)
    return simplified

# ============================================
# Step 7: Validation Stub
# ============================================
def validate_clauses(originals, simplifieds):
    validations = []
    for o, s in zip(originals, simplifieds):
        validations.append({
            "original": o,
            "simplified": s,
            "validation": "Requires legal review for compliance."
        })
    return validations

# ============================================
# Step 8: Main Pipeline
# ============================================
def process_contract(text):
    clauses = segment_document(text)
    key_clauses = identify_key_clauses(clauses)
    if not key_clauses:
        print("No key clauses matched the keywords.")
        return []
    simplifier = load_simplifier_model()
    simplified = simplify_clauses(key_clauses, simplifier)
    validated = validate_clauses(key_clauses, simplified)
    return validated

# ============================================
# Step 9: Driver Code (Main Execution)
# ============================================
if __name__ == "__main__":
    # Sample contract text (you can replace this or use file input)
    sample_contract = """
    Clause 1.1 The Parties agree that this Agreement shall be governed by the laws of the State of California.
    Clause 2.3 The Party shall indemnify and hold harmless the other Party from any liabilities, losses, and damages arising from this Agreement.
    Clause 4.2 In the event of termination, either party must give 30 days’ written notice.
    Clause 5.1 This Agreement shall remain confidential and not be disclosed to third parties without prior written consent.
    Clause 6.1 The payment terms shall be net 30 days from the date of invoice.
    Clause 7.5 Any disputes arising shall be subject to jurisdiction in San Francisco, California.
    """

    print("Processing contract...\n")
    results = process_contract(sample_contract)

    if results:
        for i, res in enumerate(results, 1):
            print(f"Clause {i}")
            print("Original   :", res['original'])
            print("Simplified :", res['simplified'])
            print("Validation :", res['validation'])
            print("-" * 80)
    else:
        print("No relevant clauses found.")