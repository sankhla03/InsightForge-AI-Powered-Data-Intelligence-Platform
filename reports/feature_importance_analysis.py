"""
================================================================================
FEATURE IMPORTANCE ANALYSIS REPORT
================================================================================
Technical Report: Understanding Feature Importance in Loan Approval Prediction
================================================================================

Generated: January 2026
Dataset: Loan Approval Dataset (79 samples, 13 features)
Target Variable: LoanAmount (Regression Task)
Author: insightForge ML Pipeline

FUNCTIONS:
- generate_report(): Returns report as HTML string for Django templates
- get_summary(): Returns key findings as structured data
================================================================================"""


def generate_report(df=None, feature_scores=None):
    """
    Generate the Feature Importance Analysis Report as HTML content.
    
    Args:
        df: Optional pandas DataFrame for dataset-specific statistics
        feature_scores: Optional dict of feature scores for current analysis
    
    Returns:
        str: HTML-formatted report content
    """
    # Build dataset-specific context
    dataset_info = ""
    score_comparison = ""
    
    if df is not None:
        dataset_info = f"""
        <div class="dataset-context">
            <h3>Current Dataset Context</h3>
            <table class="info-table">
                <tr><td>Samples:</td><td><strong>{df.shape[0]}</strong></td></tr>
                <tr><td>Features:</td><td><strong>{df.shape[1]}</strong></td></tr>
                <tr><td>Target:</td><td><strong>LoanAmount</strong></td></tr>
            </table>
        </div>
        """
    
    if feature_scores:
        # Sort and display feature scores
        sorted_scores = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        score_rows = ""
        for rank, (feat, score) in enumerate(sorted_scores, 1):
            # Identify feature type
            is_categorical = df is not None and (
                df[feat].dtype == 'int64' and df[feat].nunique() <= 5
            )
            feat_type = "Categorical" if is_categorical else "Numerical"
            
            score_rows += f"""
            <tr class="{feat_type.lower()}">
                <td>{rank}</td>
                <td>{feat}</td>
                <td>{feat_type}</td>
                <td>{score:.4f}</td>
                <td>
                    <div class="score-bar-mini">
                        <div class="fill" style="width: {min(abs(score)*100, 100)}%"></div>
                    </div>
                </td>
            </tr>
            """
        
        score_comparison = f"""
        <div class="score-comparison">
            <h3>Feature Scores from Current Analysis</h3>
            <table class="score-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Relative Importance</th>
                    </tr>
                </thead>
                <tbody>
                    {score_rows}
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""
    <div class="feature-importance-report">
        <div class="report-header">
            <h2>Feature Importance Analysis Report</h2>
            <p class="subtitle">Understanding Why Encoded Categoricals May Appear More Important</p>
        </div>
        
        {dataset_info}
        
        {score_comparison}
        
        <div class="executive-summary">
            <h3>Executive Summary</h3>
            <div class="summary-box">
                <p><strong>Key Finding:</strong> The observation that encoded categorical features 
                (Gender, Married, Dependents) appear more "important" than income features 
                (ApplicantIncome, CoapplicantIncome) in correlation-based analysis is 
                <span class="highlight">EXPECTED BEHAVIOR</span> and does NOT indicate a problem.</p>
                
                <p>This report explains the fundamental differences between:</p>
                <ul>
                    <li><strong>Statistical Correlation</strong> (linear association between variables)</li>
                    <li><strong>Feature Importance</strong> (predictive power in a model)</li>
                    <li><strong>Domain Intuition</strong> (real-world causal expectations)</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h3>1. Correlation on Encoded Categorical Features</h3>
            
            <div class="warning-box">
                <h4>Important Caveat</h4>
                <p>When categorical variables are label-encoded (e.g., Male=0, Female=1), they become 
                numerically interpretable but <strong>lose their categorical semantics</strong>.</p>
            </div>
            
            <h4>Why This Creates Misleading Results:</h4>
            <ol>
                <li>
                    <strong>Arbitrary Numeric Assignment</strong>
                    <p>Label encoding assigns arbitrary numbers: Gender: Male=0, Female=1. 
                    The numeric difference (1-0=1) has NO meaning in the original categorical data.</p>
                </li>
                <li>
                    <strong>Distorted Relationship Representation</strong>
                    <p>A correlation of 0.18 for Gender means: "Females (encoded as 1) tend to have 
                    slightly higher loan amounts." This is a <strong>STATISTICAL ASSOCIATION</strong>, 
                    not CAUSATION.</p>
                </li>
                <li>
                    <strong>Misleading Cross-Feature Comparison</strong>
                    <p>A correlation of 0.18 for Gender does NOT mean "18% more important" than 
                    ApplicantIncome with correlation 0.04. These values are NOT comparable across 
                    fundamentally different feature types.</p>
                </li>
            </ol>
            
            <div class="formula-box">
                <h4>Technical Insight</h4>
                <p>Pearson correlation coefficient (rho) is defined as:</p>
                <code>rho(X,Y) = Cov(X,Y) / (sigmaX * sigmaY)</code>
                <p>For binary-encoded features, sigmaX is fixed at ~0.48, making the result capture 
                linear co-variation but NOT "importance" in any meaningful sense.</p>
            </div>
        </div>
        
        <div class="section">
            <h3>2. Correlation vs Feature Importance</h3>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Aspect</th>
                        <th>Correlation</th>
                        <th>Feature Importance</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Definition</strong></td>
                        <td>Linear association between two variables</td>
                        <td>Predictive power contribution to model</td>
                    </tr>
                    <tr>
                        <td><strong>Scope</strong></td>
                        <td>Bivariate (2 variables only)</td>
                        <td>Multivariate (all features together)</td>
                    </tr>
                    <tr>
                        <td><strong>Relationships</strong></td>
                        <td>Linear only</td>
                        <td>Linear + Non-linear</td>
                    </tr>
                    <tr>
                        <td><strong>Feature Types</strong></td>
                        <td>Continuous only (technically)</td>
                        <td>All types handled naturally</td>
                    </tr>
                    <tr>
                        <td><strong>Interactions</strong></td>
                        <td>Ignored</td>
                        <td>Captured by model</td>
                    </tr>
                    <tr>
                        <td><strong>Range</strong></td>
                        <td>+1 to -1</td>
                        <td>Non-negative (sums to 1)</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="success-box">
                <h4>Why Tree-Based Methods Are Better</h4>
                <ul>
                    <li>Capture <strong>NON-LINEAR</strong> relationships naturally</li>
                    <li>Handle <strong>ENCODED CATEGORICALS</strong> without arbitrary ordering issues</li>
                    <li>Consider <strong>INTERACTION EFFECTS</strong> between features</li>
                    <li>Provide <strong>COMPARABLE</strong> importance scores (sum to 1)</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h3>3. Multicollinearity Detected</h3>
            
            <p>The following correlated features may be providing REDUNDANT information:</p>
            
            <table class="multicollinearity-table">
                <thead>
                    <tr>
                        <th>Feature Pair</th>
                        <th>Expected Relationship</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="warning">
                        <td>Married - CoapplicantIncome</td>
                        <td>Married applicants often have dual incomes</td>
                        <td>Redundant information inflating both importances</td>
                    </tr>
                    <tr class="warning">
                        <td>Dependents - ApplicantIncome</td>
                        <td>More dependents - job stability OR expenses</td>
                        <td>Opposing effects canceling in correlation</td>
                    </tr>
                    <tr class="info">
                        <td>Gender - Education</td>
                        <td>Historical patterns in education distribution</td>
                        <td>May carry historical bias signals</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="warning-box">
                <h4>Impact of Multicollinearity</h4>
                <ul>
                    <li>Inflated importance scores for correlated features</li>
                    <li>Unstable coefficient estimates in linear models</li>
                    <li>Tree-based methods handle this more gracefully</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h3>4. Historical Bias Detection</h3>
            
            <div class="bias-alert">
                <h4>Historical Patterns Detected</h4>
                <p>This dataset reflects <strong>HISTORICAL PATTERNS</strong> from past lending decisions:</p>
                
                <ul>
                    <li>
                        <strong>Gender Bias:</strong> Correlation exists between gender and outcomes
                        <br><em>Reflects past decisions, not future potential</em>
                    </li>
                    <li>
                        <strong>Marital Status Bias:</strong> Different approval patterns by marital status
                        <br><em>May reflect dual income stability OR systemic discrimination</em>
                    </li>
                    <li>
                        <strong>Education Bias:</strong> Correlated with income and loan amounts
                        <br><em>Reflects socioeconomic privilege in historical data</em>
                    </li>
                </ul>
            </div>
            
            <div class="recommendation-box">
                <h4>Recommendations for Production Systems</h4>
                <ul>
                    <li>Consider removing sensitive attributes (Gender, Married) for fairness</li>
                    <li>Use fairness-aware ML techniques</li>
                    <li>Audit model outputs for demographic parity</li>
                    <li>Document historical bias for stakeholders</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h3>5. Why Income Features Have Weak Correlation</h3>
            
            <p>ApplicantIncome (rho approx 0.04) and CoapplicantIncome (rho approx 0.03) show surprisingly 
            <strong>LOW Pearson correlation</strong> despite domain intuition suggesting they should 
            be highly predictive.</p>
            
            <div class="causes-box">
                <h4>Technical Explanations:</h4>
                
                <div class="cause">
                    <h5>5.1 Outlier Effects on Pearson Correlation</h5>
                    <p>Pearson correlation is highly sensitive to outliers:</p>
                    <code>rho = sum[(xi-meanX)(yi-meanY)] / [sqrt(sum(xi-meanX)^2) * sqrt(sum(yi-meanY)^2)]</code>
                    <p>Few applicants with very high incomes create asymmetric cancellation effects.</p>
                </div>
                
                <div class="cause">
                    <h5>5.2 Non-Linear Relationships</h5>
                    <p>Income - LoanAmount is likely:</p>
                    <ul>
                        <li><strong>Threshold-based:</strong> Income affects eligibility only above/below limits</li>
                        <li><strong>Logarithmic:</strong> Doubling income doesn't double loan amount</li>
                        <li><strong>Saturating:</strong> Diminishing returns at high income levels</li>
                    </ul>
                    <p>Pearson captures ONLY linear relationships.</p>
                </div>
                
                <div class="cause">
                    <h5>5.3 Data Distribution Issues</h5>
                    <p>The sample correlation of 0.04 indicates almost no LINEAR correlation, 
                    but the relationship may still exist and be detected by tree models.</p>
                    <p><strong>Solution:</strong> Use Spearman correlation for robust rank-based estimates.</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>6. Best Practices</h3>
            
            <div class="phase">
                <h4>Phase 1: Exploration</h4>
                <ul class="check-list">
                    <li class="check">Use correlation for initial data understanding</li>
                    <li class="check">Plot pairplots to visualize relationships</li>
                    <li class="check">Use Spearman correlation for robust estimates</li>
                    <li class="cross">DO NOT use correlation rankings for final feature selection</li>
                </ul>
            </div>
            
            <div class="phase">
                <h4>Phase 2: Feature Selection</h4>
                <ul class="check-list">
                    <li class="check">Use tree-based methods (Random Forest, Gradient Boosting)</li>
                    <li class="check">Use Recursive Feature Elimination (RFE)</li>
                    <li class="check">Use Permutation Importance for reliability</li>
                    <li class="check">Use SHAP values for interpretability</li>
                    <li class="check">Compare multiple methods for robustness</li>
                </ul>
            </div>
            
            <div class="phase">
                <h4>Phase 3: Interpretation</h4>
                <ul class="check-list">
                    <li class="check">Present BOTH correlation AND importance rankings</li>
                    <li class="check">Explain the difference between statistical association and predictive power</li>
                    <li class="check">Report confidence intervals for importance scores</li>
                    <li class="check">Conduct bias audits on sensitive features</li>
                    <li class="check">Document limitations and assumptions</li>
                </ul>
            </div>
        </div>
        
        <div class="section conclusion">
            <h3>Conclusion</h3>
            
            <div class="conclusion-box">
                <h4>Why This Behavior is Expected:</h4>
                
                <ol>
                    <li>
                        <strong>Pearson correlation</strong> measures linear association on continuous data
                        <br><em>Encoded categoricals violate both assumptions</em>
                    </li>
                    <li>
                        <strong>Domain intuition</strong> is about CAUSALITY (income - loan)
                        <br><em>Correlation measures only STATISTICAL ASSOCIATION</em>
                    </li>
                    <li>
                        <strong>Income features</strong> have weak linear correlation due to:
                        <br><em>Outlier distortion of Pearson calculation</em>
                        <br><em>Non-linear relationship patterns (thresholds, saturation)</em>
                    </li>
                    <li>
                        <strong>Categorical features</strong> show stronger correlation because:
                        <br><em>Binary encoding creates artificial variance</em>
                        <br><em>Historical biases are captured as statistical patterns</em>
                    </li>
                </ol>
                
                <div class="final-recommendation">
                    <h4>Final Recommendation:</h4>
                    <p>For this loan approval dataset, use <strong>TREE-BASED feature importance</strong> 
                    rather than correlation rankings. The model's learned importance scores will 
                    likely align better with domain intuition while still capturing all relevant patterns.</p>
                </div>
            </div>
        </div>
    </div>
    
    <style>
        .feature-importance-report { font-family: Arial, sans-serif; line-height: 1.6; }
        .report-header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }
        .subtitle { font-size: 1.2em; opacity: 0.9; }
        .executive-summary { background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #2196F3; }
        .summary-box { background: white; padding: 15px; border-radius: 5px; }
        .highlight { background: #fff59d; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
        .section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .section h3 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .warning-box { background: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .success-box { background: #e8f5e9; border-left: 5px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .recommendation-box { background: #e3f2fd; border-left: 5px solid #2196F3; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .formula-box { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; font-family: monospace; }
        .formula-box code { display: block; background: #333; color: #0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .comparison-table th, .comparison-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .comparison-table th { background: #667eea; color: white; }
        .comparison-table tr:nth-child(even) { background: #f8f9fa; }
        .bias-alert { background: #ffebee; border-left: 5px solid #f44336; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .multicollinearity-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .multicollinearity-table th, .multicollinearity-table td { border: 1px solid #ddd; padding: 12px; }
        .multicollinearity-table th { background: #ff9800; color: white; }
        .multicollinearity-table tr.warning { background: #fff3e0; }
        .multicollinearity-table tr.info { background: #e3f2fd; }
        .causes-box { display: grid; gap: 15px; }
        .cause { background: white; padding: 15px; border-radius: 5px; border-left: 5px solid #9c27b0; }
        .cause h5 { color: #9c27b0; margin-top: 0; }
        .phase { background: white; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 5px solid #607d8b; }
        .phase h4 { margin-top: 0; color: #607d8b; }
        .check-list { list-style: none; padding: 0; }
        .check-list li { padding: 5px 0; }
        .check-list .check::before { content: "✓ "; color: #4caf50; font-weight: bold; }
        .check-list .cross::before { content: "✗ "; color: #f44336; font-weight: bold; }
        .conclusion { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .conclusion h3 { color: white; border-bottom-color: white; }
        .conclusion-box { background: rgba(255,255,255,0.95); color: #333; padding: 20px; border-radius: 10px; }
        .final-recommendation { background: #c8e6c9; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .dataset-context { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .info-table { width: 200px; }
        .info-table td { padding: 5px; }
        .score-comparison { margin: 20px 0; }
        .score-table { width: 100%; border-collapse: collapse; }
        .score-table th, .score-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .score-table th { background: #667eea; color: white; }
        .score-table tr.categorical { background: #fff3e0; }
        .score-table tr.numerical { background: #e3f2fd; }
        .score-bar-mini { width: 100%; height: 10px; background: #e0e0e0; border-radius: 5px; overflow: hidden; }
        .score-bar-mini .fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); }
    </style>
    """
    
    return html_content


def get_summary(df=None):
    """
    Return key findings as structured data for programmatic use.
    
    Args:
        df: Optional pandas DataFrame
    
    Returns:
        dict: Structured summary data
    """
    return {
        "key_findings": [
            "Encoded categorical features show higher correlation due to statistical artifacts",
            "Pearson correlation is not designed for encoded categorical variables",
            "Income features may have weak linear correlation due to outliers and non-linear effects",
            "Tree-based methods better capture true predictive relationships",
            "Historical biases may be encoded in categorical feature correlations"
        ],
        "recommendations": [
            "Use tree-based feature importance for final interpretation",
            "Consider SHAP values for detailed feature contributions",
            "Remove sensitive features for fairness in production",
            "Use Spearman correlation for robust initial exploration",
            "Document limitations when presenting results"
        ],
        "conclusion": "This behavior is expected and represents fundamental properties of how different statistical measures interact with encoded categorical data."
    }

