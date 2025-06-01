import pandas as pd
import numpy as np
from docx import Document
import tempfile
import os
import re
from typing import List, Dict, Any, Union, Optional
import io
import matplotlib.pyplot as plt
#import seaborn as sns

class FileProcessor:
    """Utility class for handling file operations"""
    
    @staticmethod
    def read_excel_file(file) -> pd.DataFrame:
        """
        Read Excel or CSV file and return DataFrame
        """
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError(f"Unsupported file format: {file.name}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            return df
            
        except Exception as e:
            raise Exception(f"Error reading file {file.name}: {str(e)}")
    
    @staticmethod
    def save_temp_docx(uploaded_file) -> str:
        """
        Save uploaded Word document to temporary file and return path
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """
        Clean up temporary file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {file_path}: {e}")

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_supplier_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate supplier data DataFrame and return validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        required_columns = ['Supplier', 'OnTimeDeliveryRate', 'DefectRate']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check data types and ranges
        if 'OnTimeDeliveryRate' in df.columns:
            invalid_rates = df[
                (df['OnTimeDeliveryRate'] < 0) | (df['OnTimeDeliveryRate'] > 100)
            ]
            if not invalid_rates.empty:
                validation_results["warnings"].append(
                    f"OnTimeDeliveryRate should be between 0-100. Found {len(invalid_rates)} invalid values."
                )
        
        if 'DefectRate' in df.columns:
            invalid_defects = df[(df['DefectRate'] < 0) | (df['DefectRate'] > 100)]
            if not invalid_defects.empty:
                validation_results["warnings"].append(
                    f"DefectRate should be between 0-100. Found {len(invalid_defects)} invalid values."
                )
        
        # Summary statistics
        validation_results["summary"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_suppliers": df.duplicated(subset=['Supplier']).sum() if 'Supplier' in df.columns else 0,
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return validation_results
    
    @staticmethod
    def clean_supplier_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess supplier data
        """
        df_clean = df.copy()
        
        # Remove leading/trailing whitespace from string columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove duplicate suppliers (keep first occurrence)
        if 'Supplier' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['Supplier'], keep='first')
        
        return df_clean

class AuditProcessor:
    """Utility class for processing audit documents"""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """
        Extract text from Word document
        """
        try:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error reading document: {str(e)}")
    
    @staticmethod
    def extract_supplier_name(content: str, filename: str) -> str:
        """
        Extract supplier name from document content or filename
        """
        # Try to find supplier name in content using regex
        patterns = [
            r'SUPPLIER\s+(\d+)S?',  # SUPPLIER 1S, SUPPLIER 2, etc.
            r'Supplier\s+(\d+)',    # Supplier 1, Supplier 2, etc.
            r'supplier\s+(\d+)',    # supplier 1, supplier 2, etc.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return f"Supplier {match.group(1)}"
        
        # Fallback to filename parsing
        filename_patterns = [
            r'Audit.*Report.*?(\d+)',  # Audit_Report_1.docx, etc.
            r'supplier.*?(\d+)',       # supplier_1_audit.docx, etc.
            r'audit.*?(\d+)',          # audit_1.docx, etc.
        ]
        
        for pattern in filename_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"Supplier {match.group(1)}"
        
        # Final fallback
        return f"Supplier from {filename}"
    
    @staticmethod
    def count_audit_findings(content: str) -> int:
        """
        Count audit findings in document content using multiple approaches
        """
        content_lower = content.lower()
        
        # Initialize count
        total_findings = 0
        
        # Method 1: Look for explicit numbers
        explicit_patterns = [
            r'(\d+)\s+major\s+finding',
            r'(\d+)\s+minor\s+finding',
            r'(\d+)\s+finding',
            r'no\s+findings?\s+observed',
            r'zero\s+findings?'
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                if 'no' in pattern or 'zero' in pattern:
                    return 0
                for match in matches:
                    total_findings += int(match)
        
        # Method 2: Count individual mentions if no explicit numbers found
        if total_findings == 0:
            finding_keywords = [
                'major finding', 'minor finding', 'critical finding',
                'observation', 'non-conformance', 'deficiency'
            ]
            
            for keyword in finding_keywords:
                count = content_lower.count(keyword)
                # Weight different types of findings
                if 'major' in keyword or 'critical' in keyword:
                    total_findings += count * 2  # Major findings count double
                else:
                    total_findings += count
        
        # Method 3: Special case handling
        if 'no findings' in content_lower:
            return 0
        
        return total_findings
    
    @staticmethod
    def extract_audit_metadata(content: str) -> Dict[str, Any]:
        """
        Extract metadata from audit document
        """
        metadata = {
            'audit_type': 'Unknown',
            'location': 'Unknown',
            'auditor': 'Unknown',
            'date': 'Unknown',
            'iso_standard': None
        }
        
        # Extract audit type
        if re.search(r'surveillance\s+audit', content, re.IGNORECASE):
            metadata['audit_type'] = 'Surveillance'
        elif re.search(r'certification\s+audit', content, re.IGNORECASE):
            metadata['audit_type'] = 'Certification'
        
        # Extract location
        location_match = re.search(r'located\s+in\s+([^.]+)', content, re.IGNORECASE)
        if location_match:
            metadata['location'] = location_match.group(1).strip()
        
        # Extract ISO standard
        iso_match = re.search(r'ISO\s+(\d+:\d+)', content)
        if iso_match:
            metadata['iso_standard'] = iso_match.group(1)
        
        return metadata

class ChartGenerator:
    """Utility class for generating charts and visualizations"""
    
    @staticmethod
    def create_performance_chart(scores: Dict[str, float], 
                               title: str = "Supplier Performance Scores",
                               figsize: tuple = (10, 6)) -> str:
        """
        Create a performance chart and return the file path
        """
        plt.figure(figsize=figsize)
        
        # Sort scores for better visualization
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
        bars = plt.bar(sorted_scores.keys(), sorted_scores.values(), 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel("Performance Score", fontsize=12)
        plt.xlabel("Suppliers", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save to temporary file
        chart_path = tempfile.mktemp(suffix='.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    @staticmethod
    def create_distribution_chart(data: Dict[str, Any], 
                                title: str = "Distribution Analysis") -> str:
        """
        Create distribution charts for various metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Example: You can expand this based on your data structure
        # This is a template that can be customized
        
        chart_path = tempfile.mktemp(suffix='.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

class ScoreCalculator:
    """Utility class for calculating various performance scores"""
    
    @staticmethod
    def calculate_weighted_score(metrics: Dict[str, float], 
                               weights: Dict[str, float]) -> float:
        """
        Calculate weighted performance score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                total_score += value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def normalize_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
        """
        Normalize score to a specific range
        """
        return max(min_val, min(max_val, score))
    
    @staticmethod
    def calculate_performance_tier(score: float) -> str:
        """
        Determine performance tier based on score
        """
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Satisfactory"
        elif score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"

class ReportFormatter:
    """Utility class for formatting reports and outputs"""
    
    @staticmethod
    def format_score_table(scores: Dict[str, float]) -> pd.DataFrame:
        """
        Format scores into a nicely structured DataFrame
        """
        df = pd.DataFrame(list(scores.items()), columns=['Supplier', 'Score'])
        df = df.sort_values('Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        df['Performance Tier'] = df['Score'].apply(ScoreCalculator.calculate_performance_tier)
        df['Score'] = df['Score'].round(2)
        
        return df[['Rank', 'Supplier', 'Score', 'Performance Tier']]
    
    @staticmethod
    def format_findings_summary(findings: Dict[str, int]) -> pd.DataFrame:
        """
        Format audit findings into a structured DataFrame
        """
        df = pd.DataFrame(list(findings.items()), 
                         columns=['Supplier', 'Findings Count'])
        df['Risk Level'] = df['Findings Count'].apply(
            lambda x: 'High' if x > 3 else 'Medium' if x > 1 else 'Low'
        )
        
        return df.sort_values('Findings Count', ascending=False)

# Helper functions for common operations
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    """
    return numerator / denominator if denominator != 0 else default

def extract_numeric_value(text: str, default: float = 0.0) -> float:
    """
    Extract numeric value from text string
    """
    if pd.isna(text):
        return default
    
    # Try to extract number from string
    numbers = re.findall(r'\d+\.?\d*', str(text))
    if numbers:
        return float(numbers[0])
    
    return default

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    """
    if pd.isna(text):
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', str(text).strip())
    return cleaned
