# remera_dashboard.py
# Author: Mutabazi Samuel
# Project: Interactive Dashboard for Trained Citizens in Remera Cell

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime  # NECESSARY for calculate_age_from_id
import warnings
from io import BytesIO  # For generating report content

# Suppress warnings related to Matplotlib and chained assignment (Pandas)
warnings.filterwarnings('ignore')

# ==========================
# 1. PAGE CONFIGURATION
# ==========================
# Keeps the dashboard wide across the screen
st.set_page_config(page_title="Remera Community Training Dashboard", layout="wide")

st.title("📊 Remera Community Training Dashboard")
# Dynamic description placeholder
initial_description = "Welcome to the interactive data dashboard for trained citizens. Please upload your file to begin analysis."
st_description_placeholder = st.empty()
st_description_placeholder.write(initial_description)


# --- NID Age Calculation Function ---
@st.cache_data
def calculate_age_from_id(national_id_number: str) -> int | None:
	"""
	Calculates age based on a 16-digit Rwandan NID (assuming 1/2 prefix and YYMMDD).
	This function is cached to speed up calculations.
	"""
	# Ensure it is a valid 16-digit string
	if not isinstance(national_id_number, str) or len(national_id_number) != 16 or not national_id_number.isdigit():
		return None

	try:
		century_prefix = national_id_number[0]
		if century_prefix == '1':
			century_start = 1900
		elif century_prefix == '2':
			century_start = 2000
		else:
			return None  # Invalid century prefix

		# Extract Birth Year (YY) from digits 7 and 8 (index 6 to 8)
		year_suffix = national_id_number[6:8]
		birth_year = century_start + int(year_suffix)  # Correctly calculate the 4-digit year

		current_year = datetime.date.today().year
		age = current_year - birth_year

		# Basic sanity check for age
		return age if 0 < age < 120 else None

	except Exception:
		return None


# ==========================
# 2. LOAD DATA
# ==========================
uploaded_file = st.file_uploader("📂 Upload your Data file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
	# --- 2.1 Load Data ---
	file_extension = uploaded_file.name.split('.')[-1].lower()

	try:
		if file_extension == 'csv':
			df = pd.read_csv(uploaded_file)
			file_message = "CSV"
		elif file_extension in ['xlsx', 'xls']:
			df = pd.read_excel(uploaded_file)
			file_message = "Excel"
		else:
			st.error("❌ Unsupported file type uploaded.")
			st.stop()

		st.success(f"✅ {file_message} file uploaded successfully! Original size: {len(df)} rows.")

		# Store original size for data quality check later
		st.session_state['original_data_size'] = len(df)

	except Exception as e:
		st.error(f"❌ Error loading data from file: {e}")
		st.stop()

	# ==========================
	# 3. CLEAN & PREPARE DATA
	# ==========================

	# --- 3.1 Column Renaming & Standardization ---
	rename_map = {
		'date of training': 'Training Date', 'training_date': 'Training Date', 'date': 'Training Date',
		'sex': 'Gender', 'gender ': 'Gender',
		'AGE': 'Age', 'age ': 'Age',
		'ID': 'National ID',
		'district': 'District', 'province': 'District',
		'sector': 'Sector', 'secteur': 'Sector',
		'cell': 'Cell', 'umudugudu': 'Cell',
		'disability': 'Disability Status', 'disabled': 'Disability Status'
	}

	df.columns = [c.strip() for c in df.columns]
	lower_map = {c.lower(): c for c in df.columns}
	# Apply renaming based on lowercase matching
	for key, new_name in rename_map.items():
		if key.lower() in lower_map:
			df.rename(columns={lower_map[key.lower()]: new_name}, inplace=True)

	# Check if 'National ID' column exists after rename (or if it was already named 'ID')
	if 'National ID' not in df.columns and 'ID' in df.columns:
		df.rename(columns={'ID': 'National ID'}, inplace=True)

	st.write("🧾 **Detected Columns After Cleanup:**", list(df.columns))

	# --- NEW: NID Cleaning to ensure format compliance ---
	if 'National ID' in df.columns:
		# Convert to string, strip whitespace, and remove all non-digit characters
		df['National ID'] = df['National ID'].astype(str).str.strip().str.replace(r'\D', '', regex=True)

	# --- 3.2 Column Presence Check ---
	date_col_present = 'Training Date' in df.columns
	age_col_present = 'Age' in df.columns
	gender_col_present = 'Gender' in df.columns
	nid_col_present = 'National ID' in df.columns
	disability_col_present = 'Disability Status' in df.columns
	district_col_present = 'District' in df.columns
	sector_col_present = 'Sector' in df.columns
	cell_col_present = 'Cell' in df.columns

	# Critical column checks
	if not date_col_present: st.error("⚠️ Missing 'Training Date' column — cannot proceed."); st.stop()

	# Convert types
	df['Training Date'] = pd.to_datetime(df['Training Date'], errors='coerce')
	df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

	# --- 3.3 Final Cleaning (The Critical Step) ---
	# Drop rows where Training Date failed to convert (is NaT)
	base_df = df.dropna(subset=['Training Date'], how='any').copy()

	if base_df.empty:
		st.error(
			"🚫 **DATA CRITICAL ERROR:** All rows were dropped because they were missing valid 'Training Date' data. Please check your file.")
		st.stop()

	# Store size after initial cleaning for data quality report
	st.session_state['cleaned_data_size'] = len(base_df)


	# --- 3.4 Feature Engineering & Age Calculation ---
	def age_group(age):
		if pd.isna(age):
			return 'Unknown'
		elif age < 18:
			return 'Under 18'
		elif age <= 30:
			return '18–30'
		elif age <= 45:
			return '31–45'
		elif age <= 60:
			return '46–60'
		else:
			return 'Above 60'


	# Initial Age Grouping based on raw Age column (may contain NaNs)
	base_df['Age Group'] = base_df['Age'].apply(age_group)
	base_df['Training Month'] = base_df['Training Date'].dt.strftime('%Y-%m')  # Use YYYY-MM for sorting

	# ==========================
	# 4. SIDEBAR CONTROLS
	# ==========================
	st.sidebar.header("Filter Data")

	# --- NID Age Calculation Section (Integration of Logic) ---
	st.sidebar.subheader("National ID Age Calculation")
	nid_column_name_input = "National ID"  # Fixed input after initial setup

	# If the NID column is present, run calculation and update
	if nid_col_present:
		# 1. Calculate new age
		base_df['Age Calculated'] = base_df[nid_column_name_input].astype(str).apply(calculate_age_from_id)

		# 2. Update the 'Age' column using Age Calculated where it's not null.
		# Using numpy.where is a fast, vectorized, and safe way to prevent SettingWithCopyWarning.
		base_df['Age'] = np.where(
			base_df['Age Calculated'].notna(),
			base_df['Age Calculated'],
			base_df['Age']
		)

		# 3. Re-run age grouping based on the final Age column
		base_df['Age Group'] = base_df['Age'].apply(age_group)

		# Provide more detailed feedback
		successful_updates = base_df['Age Calculated'].count()
		st.sidebar.success(f"✅ Age updated using NID. ({successful_updates} successful updates)")
	else:
		st.sidebar.warning("National ID column not found. Using raw Age data.")

	# Filter cleanup: Drop rows where age is still invalid after NID calculation
	base_df = base_df.dropna(subset=['Age']).copy()

	# --- GEOGRAPHICAL FILTERS (B) ---
	st.sidebar.subheader("Geographical Filters")
	filtered_df = base_df.copy()

	if district_col_present:
		district_options = base_df['District'].dropna().unique()
		district_filter = st.sidebar.multiselect("Select District", options=district_options, default=district_options)
		filtered_df = filtered_df[filtered_df['District'].isin(district_filter)]

	if sector_col_present:
		sector_options = filtered_df['Sector'].dropna().unique()
		sector_filter = st.sidebar.multiselect("Select Sector", options=sector_options, default=sector_options)
		filtered_df = filtered_df[filtered_df['Sector'].isin(sector_filter)]

	if cell_col_present:
		cell_options = filtered_df['Cell'].dropna().unique()
		cell_filter = st.sidebar.multiselect("Select Cell", options=cell_options, default=cell_options)
		filtered_df = filtered_df[filtered_df['Cell'].isin(cell_filter)]

	# --- DEMOGRAPHIC FILTERS ---
	st.sidebar.subheader("Demographic Filters")

	if gender_col_present:
		gender_options = filtered_df['Gender'].dropna().unique()
		gender_filter = st.sidebar.multiselect("Select Gender", options=gender_options, default=gender_options)
		filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]

	age_options = filtered_df['Age Group'].unique()
	age_filter = st.sidebar.multiselect("Select Age Group", options=age_options, default=age_options)
	filtered_df = filtered_df[filtered_df['Age Group'].isin(age_filter)]

	if disability_col_present:
		disability_options = filtered_df['Disability Status'].dropna().unique()
		disability_filter = st.sidebar.multiselect("Select Disability Status", options=disability_options,
												   default=disability_options)
		filtered_df = filtered_df[filtered_df['Disability Status'].isin(disability_filter)]

	# =======================================================
	# 5. ANALYTICS & VISUALS (Only runs if data exists)
	# =======================================================
	if not filtered_df.empty:
		# Dynamic description update
		unique_months = filtered_df['Training Month'].nunique()
		description_text = f"Welcome to the interactive data dashboard for trained citizens. Your dataset covers **{unique_months} unique training months**."
		st_description_placeholder.write(description_text)

		st.subheader("📈 Data Analysis and Visualization")

		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Total Citizens Trained", len(filtered_df))
		with col2:
			st.metric("Unique Training Months", unique_months)
		with col3:
			st.metric("Average Age", round(filtered_df['Age'].mean(), 1))

		# --- 5.1 AUTOMATIC INSIGHTS PANEL (A) ---
		st.markdown("---")
		st.subheader("💡 Automatic Insights Panel")

		# Calculate simple insights
		total_trainees = len(filtered_df)
		avg_age = filtered_df['Age'].mean()
		peak_month = filtered_df['Training Month'].mode()[0] if not filtered_df[
			'Training Month'].mode().empty else "N/A"

		insight_text = f"""
       * **Overall Reach:** The dashboard is currently analyzing **{total_trainees}** citizens across **{unique_months}** training periods.
       * **Average Trainee:** The average age of trained citizens is **{avg_age: .1f}** years old, falling generally within the 18–45 age brackets.
       * **Peak Activity:** The month with the highest training volume was **{peak_month}**.
       """
		if gender_col_present:
			gender_ratio = filtered_df['Gender'].value_counts(normalize=True).max()
			dominant_gender = filtered_df['Gender'].value_counts().idxmax()
			insight_text += f"\n* **Gender Focus:** **{dominant_gender}** represents **{gender_ratio * 100:.1f}%** of all trainees. Consider strategies to balance representation if necessary."

		st.info(insight_text)
		st.markdown("---")

		# Row 2: Distribution Charts
		col4, col5 = st.columns(2)
		with col4:
			# Gender distribution (Pie Chart)
			if gender_col_present and not filtered_df['Gender'].empty:
				gender_counts = filtered_df['Gender'].value_counts()
				fig1, ax1 = plt.subplots()
				ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
				ax1.set_title('Gender Distribution')
				st.pyplot(fig1)
		with col5:
			# Age group distribution (Pie Chart)
			age_counts = filtered_df['Age Group'].value_counts()
			fig2, ax2 = plt.subplots()
			ax2.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
			ax2.set_title('Age Group Distribution')
			st.pyplot(fig2)

		# --- 5.2 MONTHLY TRAINING TREND & FORECAST (D) ---
		st.subheader("Monthly Training Trend & Forecast")

		# Calculate absolute counts per month
		month_counts = filtered_df.groupby('Training Month').size()

		# Convert YYYY-MM labels to numerical indices for forecasting
		x_hist = np.arange(len(month_counts))
		y_hist = month_counts.values

		# Perform linear fit (Trend Forecast without ML)
		if len(month_counts) >= 2:
			# Fit a 1st degree polynomial (straight line)
			coefficients = np.polyfit(x_hist, y_hist, 1)
			polynomial = np.poly1d(coefficients)

			# Forecast for the next 2 time steps (2 months)
			x_forecast = np.arange(len(month_counts) + 2)
			y_forecast = polynomial(x_forecast)

			# Create forecast labels
			last_month = datetime.datetime.strptime(month_counts.index[-1], '%Y-%m')
			forecast_months = [
				(last_month + pd.DateOffset(months=1)).strftime('%Y-%m'),
				(last_month + pd.DateOffset(months=2)).strftime('%Y-%m')
			]
			full_labels = list(month_counts.index) + forecast_months

		else:
			# If data is too short for a reliable forecast
			y_forecast = y_hist
			full_labels = list(month_counts.index)

		fig3, ax3 = plt.subplots(figsize=(14, 7))

		# Plot historical data
		ax3.bar(month_counts.index, month_counts.values, color='forestgreen', label='Actual Trainees')

		# Plot forecast data if available
		if len(month_counts) >= 2:
			# Plot trend line over full period
			ax3.plot(full_labels, y_forecast, color='red', linestyle='--', label='Linear Trend Forecast')

			# Highlight the forecast points
			ax3.scatter(forecast_months, y_forecast[-2:], color='red', marker='o', s=100, zorder=5)

			# Add text labels for forecast points
			for i, txt in enumerate(y_forecast[-2:]):
				ax3.annotate(f"{int(round(txt))}", (forecast_months[i], txt), textcoords="offset points",
							 xytext=(0, 10), ha='center', color='red', weight='bold')

			st.info(
				f"🔮 **Forecast:** The simple linear trend predicts **{int(round(y_forecast[-2]))}** trainees next month ({forecast_months[0]}) and **{int(round(y_forecast[-1]))}** the month after ({forecast_months[1]}).")

		ax3.set_title('Absolute Number of Citizens Trained per Month (with Trend Forecast)')
		ax3.set_xlabel('Training Month (YYYY-MM)')
		ax3.set_ylabel('Total Citizens Trained')
		plt.xticks(rotation=45)
		ax3.legend()
		plt.grid(axis='y', linestyle='--')
		plt.tight_layout()
		st.pyplot(fig3)

		# --- 5.3 GEOGRAPHICAL ANALYSIS CHART (B) ---
		st.subheader("Geographical Distribution Analysis")
		geo_choice = st.radio("Select Level of Analysis:", ('Cell', 'Sector', 'District'))

		geo_col_present = {
			'Cell': cell_col_present,
			'Sector': sector_col_present,
			'District': district_col_present
		}

		if geo_col_present.get(geo_choice, False):
			geo_counts = filtered_df[geo_choice].value_counts().sort_values(ascending=False)
			fig6, ax6 = plt.subplots(figsize=(14, 7))
			geo_counts.plot(kind='bar', ax=ax6, colormap='viridis')

			ax6.set_title(f'Trainee Count by {geo_choice}')
			ax6.set_xlabel(geo_choice)
			ax6.set_ylabel('Total Trainees')
			plt.xticks(rotation=45, ha='right')
			plt.tight_layout()
			st.pyplot(fig6)
		else:
			st.warning(f"⚠️ The '{geo_choice}' column was not found in the uploaded file, or has been filtered out.")

		# --- Composition Charts ---
		st.subheader("Composition Analysis (Age and Gender Composition by Month)")

		# Age Group Stacked Bar
		composition_data_age = filtered_df.groupby('Training Month')['Age Group'].value_counts().unstack(fill_value=0)
		composition_data_age = composition_data_age.div(composition_data_age.sum(axis=1), axis=0) * 100

		fig4, ax4 = plt.subplots(figsize=(14, 7))  # INCREASED SIZE
		composition_data_age.plot(kind='bar', stacked=True, ax=ax4, colormap='Spectral')

		ax4.set_title('Training Cohort Composition (Age Group % Breakdown by Month)')
		ax4.set_xlabel('Training Month')
		ax4.set_ylabel('Percentage of Trainees (%)')
		ax4.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.xticks(rotation=45)
		plt.tight_layout()
		st.pyplot(fig4)

		# Gender Stacked Bar
		if gender_col_present and not filtered_df['Gender'].empty:
			composition_data_gender = filtered_df.groupby('Training Month')['Gender'].value_counts().unstack(
				fill_value=0)
			composition_data_gender = composition_data_gender.div(composition_data_gender.sum(axis=1), axis=0) * 100

			fig5, ax5 = plt.subplots(figsize=(14, 7))  # INCREASED SIZE
			composition_data_gender.plot(kind='bar', stacked=True, ax=ax5, colormap='Pastel1')

			ax5.set_title('Training Gender Composition (% Breakdown by Month)')
			ax5.set_xlabel('Training Month')
			ax5.set_ylabel('Percentage of Trainees (%)')
			ax5.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
			plt.xticks(rotation=45)
			plt.tight_layout()
			st.pyplot(fig5)

		# ==========================
		# 7. DATA QUALITY, REPORT & EXPORT SECTION
		# ==========================
		st.subheader("Data Quality, Reporting, and Export")

		# --- 7.1 DATA QUALITY DIAGNOSTICS (C) ---
		with st.expander("📊 Data Quality Diagnostics"):
			original_size = st.session_state.get('original_data_size', 0)
			cleaned_size = len(base_df)

			st.metric("Total Rows Processed", f"{original_size} rows")
			st.metric("Rows Dropped (Invalid Date/Age)", f"{original_size - cleaned_size} rows")

			st.markdown("**Missing Data Analysis (on cleaned data):**")

			key_cols = ['Training Date', 'Age', 'Gender', 'National ID', 'Disability Status', 'District', 'Sector',
						'Cell']

			missing_data = []
			for col in key_cols:
				if col in base_df.columns:
					missing_count = base_df[col].isnull().sum()
					missing_percent = (missing_count / cleaned_size) * 100 if cleaned_size > 0 else 0
					missing_data.append(
						{'Column': col, 'Missing Count': missing_count, 'Missing %': f"{missing_percent:.2f}%"})

			st.dataframe(pd.DataFrame(missing_data).set_index('Column'), use_container_width=True)
			st.caption("Missing data is calculated before applying geographical/demographic filters.")

		# --- 7.2 PDF REPORT GENERATOR (E) ---
		st.subheader("Generate Filtered PDF Report")


		def generate_report_markdown(df_metrics, df_geo, df_quality, insight_text):
			report_md = f"""# Remera Training Data Report

**Report Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Filters Applied:** Yes (data reflects current selections)

---

## 1. Executive Summary & Insights
{insight_text}

---

## 2. Key Metrics
| Metric | Value |
| :--- | :--- |
{df_metrics.to_markdown(index=False)}

---

## 3. Geographical Breakdown
**Training Count by {geo_choice}:**
{df_geo.to_markdown(index=False)}

---

## 4. Data Quality Summary
| Column | Missing Count | Missing % |
| :--- | :--- | :--- |
{df_quality.to_markdown(index=False)}

---

## 5. Raw Data Sample
Showing the first 10 rows of the filtered dataset:
{filtered_df.head(10).to_markdown(index=False)}

*Note: Visualizations (charts) are best viewed in the interactive dashboard.*
"""
			return report_md.encode('utf-8')


		# Prepare data frames for report
		metrics_data = pd.DataFrame({
			'Metric': ['Total Trainees', 'Unique Months', 'Average Age'],
			'Value': [len(filtered_df), unique_months, round(filtered_df['Age'].mean(), 1)]
		})

		quality_data = pd.DataFrame(missing_data)

		geo_counts_report = filtered_df[geo_choice].value_counts().reset_index()
		geo_counts_report.columns = [geo_choice, 'Trainee Count']

		report_data = generate_report_markdown(metrics_data, geo_counts_report, quality_data, insight_text)

		st.download_button(
			label="Download Report (Markdown for PDF)",
			data=report_data,
			file_name=f'Remera_Report_{datetime.date.today().strftime("%Y%m%d")}.md',
			mime='text/markdown',
			help="Download this Markdown file, then open it in any text editor and print/export it to a PDF document."
		)

		# --- Final Data Export ---
		st.subheader("💾 Export Cleaned Data")
		csv = filtered_df.to_csv(index=False).encode('utf-8')
		st.download_button(
			label="Download Cleaned CSV",
			data=csv,
			file_name='remera_data_cleaned.csv',
			mime='text/csv',
		)

	else:
		st.warning("⚠️ The current filter selection resulted in no data to display. Please adjust the filters.")


else:
	# Retain placeholder logic for when no file is uploaded
	st_description_placeholder.write(initial_description)
	st.info("👆 Please upload your CSV or Excel file to start the analysis.")
