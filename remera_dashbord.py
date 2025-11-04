# remera_dashboard.py
# Author: Mutabazi Samuel
# Project: Interactive Dashboard for Trained Citizens in Remera Cell

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 1. PAGE CONFIGURATION
# ==========================
st.set_page_config(page_title="Remera Training Data Dashboard", layout="wide")

st.title("📊 Trained Citizens in Remera Cell")
st.write(
	"Welcome to the interactive data dashboard for trained citizens during the seven months training program in Remera Cell."
)

# ==========================
# 2. LOAD DATA
# ==========================
# UPDATE: Added 'xlsx' and 'xls' to accepted file types
uploaded_file = st.file_uploader("📂 Upload your Data file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
	# --- 2.1 Load Data ---
	file_extension = uploaded_file.name.split('.')[-1].lower()

	try:
		if file_extension == 'csv':
			df = pd.read_csv(uploaded_file)
			file_message = "CSV"
		elif file_extension in ['xlsx', 'xls']:
			# Use pd.read_excel for Excel files
			df = pd.read_excel(uploaded_file)
			file_message = "Excel"
		else:
			st.error("❌ Unsupported file type uploaded.")
			st.stop()

		st.success(f"✅ {file_message} file uploaded successfully!")
		st.write("🧾 **Detected Columns:**", list(df.columns))
	except Exception as e:
		st.error(f"❌ Error loading data from file: {e}")
		st.stop()  # Stop execution if loading fails

	# ==========================
	# 3. CLEAN & PREPARE DATA
	# ==========================

	# --- 3.1 Column Renaming ---
	rename_map = {
		'date of training': 'Training Date',
		'training_date': 'Training Date',
		'date': 'Training Date',
		'sex': 'Gender',
		'gender ': 'Gender',
		'AGE': 'Age',
		'age ': 'Age',
	}

	df.columns = [c.strip() for c in df.columns]
	lower_map = {c.lower(): c for c in df.columns}
	for key, new_name in rename_map.items():
		if key in lower_map:
			df.rename(columns={lower_map[key]: new_name}, inplace=True)

	st.write("🔁 **After Renaming:**", list(df.columns))

	# --- 3.2 Type Conversion and Validation ---

	date_col_present = 'Training Date' in df.columns
	age_col_present = 'Age' in df.columns
	gender_col_present = 'Gender' in df.columns

	if date_col_present:
		# Convert with errors='coerce' to turn bad values into NaT (Not a Time)
		df['Training Date'] = pd.to_datetime(df['Training Date'], errors='coerce')
	else:
		st.error("⚠️ Missing 'Training Date' column — cannot proceed.")
		st.stop()

	if age_col_present:
		# Convert with errors='coerce' to turn bad values into NaN (Not a Number)
		df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
	else:
		st.error("⚠️ Missing 'Age' column — cannot proceed.")
		st.stop()

	if not gender_col_present:
		st.warning("⚠️ 'Gender' column not found — gender charts will be skipped.")

	# --- 3.3 Final Cleaning (The Critical Step) ---
	# Drop rows where Age or Training Date failed to convert (are NaT/NaN)
	base_df = df.dropna(subset=['Age', 'Training Date'], how='any').copy()

	# Check if data remains after cleaning
	if base_df.empty:
		st.error(
			"🚫 **DATA CRITICAL ERROR:** All rows were dropped because they were missing valid 'Age' or 'Training Date' data. Please check your uploaded CSV or Excel file for formatting issues.")
		st.stop()  # Stop execution if no data remains


	# --- 3.4 Feature Engineering ---
	# Age grouping
	def age_group(age):
		if age < 18:
			return 'Under 18'
		elif age <= 30:
			return '18–30'
		elif age <= 45:
			return '31–45'
		elif age <= 60:
			return '46–60'
		else:
			return 'Above 60'


	base_df['Age Group'] = base_df['Age'].apply(age_group)
	base_df['Training Month'] = base_df['Training Date'].dt.strftime('%B %Y')

	# ==========================
	# 4. DASHBOARD SECTIONS
	# ==========================
	st.subheader("👥 Basic Information")
	st.dataframe(base_df.head())

	# Sidebar filters
	st.sidebar.header("Filter Data")

	# Start filtering from a copy of the base data
	filtered_df = base_df.copy()

	# --- 4.1 Gender Filter ---
	if gender_col_present:
		# Get options only from the cleaned data
		gender_options = base_df['Gender'].dropna().unique()

		# Only display filter if there are gender options available
		if gender_options.size > 0:
			gender_filter = st.sidebar.multiselect(
				"Select Gender", options=gender_options,
				default=gender_options
			)
			# Apply gender filter
			filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
		else:
			st.sidebar.warning("No Gender data available to filter.")

	# --- 4.2 Age Group Filter ---
	age_options = base_df['Age Group'].unique()
	age_filter = st.sidebar.multiselect(
		"Select Age Group", options=age_options,
		default=age_options
	)
	# Apply age filter
	filtered_df = filtered_df[filtered_df['Age Group'].isin(age_filter)]

	# =======================================================
	# 5. ANALYTICS & VISUALS (Only runs if data exists)
	# =======================================================
	if not filtered_df.empty:
		st.subheader("📈 Data Analysis and Visualization")

		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Total Citizens Trained", len(filtered_df))
		with col2:
			st.metric("Unique Training Months", filtered_df['Training Month'].nunique())
		with col3:
			st.metric("Average Age", round(filtered_df['Age'].mean(), 1))

		# Gender distribution
		if gender_col_present and 'Gender' in filtered_df.columns and not filtered_df['Gender'].empty:
			gender_counts = filtered_df['Gender'].value_counts()
			fig1, ax1 = plt.subplots()
			ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
			ax1.set_title('Gender Distribution')
			st.pyplot(fig1)

		# Age group distribution
		age_counts = filtered_df['Age Group'].value_counts()
		fig2, ax2 = plt.subplots()
		ax2.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
		ax2.set_title('Age Group Distribution')
		st.pyplot(fig2)

		# Training per month
		month_counts = filtered_df['Training Month'].value_counts().sort_index()
		fig3, ax3 = plt.subplots(figsize=(10, 5))
		month_counts.plot(kind='bar', color='skyblue', ax=ax3)
		ax3.set_title('Number of Trained Citizens per Month')
		ax3.set_xlabel('Month')
		ax3.set_ylabel('Number of Trainees')
		plt.xticks(rotation=45)
		st.pyplot(fig3)

		# ==========================
		# 6. EXPORT SECTION
		# ==========================
		st.subheader("💾 Export Cleaned Data")
		csv = filtered_df.to_csv(index=False).encode('utf-8')
		st.download_button(
			label="Download Cleaned CSV",
			data=csv,
			file_name='remera_data_cleaned.csv',
			mime='text/csv',
		)

	else:
		# Warning message if the filters result in zero data
		st.warning(
			"⚠️ The current filter selection resulted in no data to display. Please adjust the Gender or Age Group filters.")


else:
	st.info("👆 Please upload your CSV or Excel file to start the analysis.")
