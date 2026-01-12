📊 Remera Community Training Dashboard
An interactive, data-driven web application designed to analyze and visualize training outcomes for citizens in the Remera Cell. This tool enables local administrators to transform raw spreadsheets into actionable insights, helping to track demographic reach and forecast future training needs.

🚀 Core Features
1. Smart Data Ingestion & Cleaning
Multi-format Support: Seamlessly handles .csv, .xlsx, and .xls files.

Auto-Standardization: Automatically detects and renames common column variations (e.g., mapping "secteur" to "Sector").

Data Quality Diagnostics: Includes an expander that tracks dropped rows and missing data percentages across 8 key variables.

2. Advanced NID Age Logic
Automated Age Extraction: Features a cached function that extracts birth years from 16-digit Rwandan National IDs (NID).

Smart Update: Automatically prioritizes NID-calculated ages over raw data to ensure 100% accuracy in demographic reporting.

3. Dynamic Analytics & Visualization
Automatic Insights Panel: Real-time generation of executive summaries, identifying peak training months and gender dominance.

Geographical Deep-Dives: Toggle analysis levels between Cell, Sector, and District.

Cohort Composition: Stacked bar charts showing the percentage breakdown of Age Groups and Gender over time.

4. Predictive Forecasting
Linear Trend Analysis: Uses polynomial fitting to predict trainee volume for the next two months based on historical trends.

Visual Annotations: Highlights forecast points with red markers and bold data labels.

5. Professional Reporting
Markdown Export: Generates a structured report including executive summaries, metrics, and data quality logs ready for PDF conversion.

Clean Data Export: One-click download of the filtered and cleaned dataset.
