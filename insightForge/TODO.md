# TODO: Ordinal Encoding UI Improvements

## Task Summary
1. Convert checkbox to button ✅
2. Add "Show Encoded Data" button in results page ✅
3. Display actual encoded dataset with all values ✅
4. **View full dataset immediately when clicking "Show Data" (without clicking "Run Feature Selection" first)** ✅
5. **Auto-save encoded dataset to session for subsequent feature selection** ✅

## Changes Made:

### preprocessing/templates/preprocessing/features.html
- Replaced checkbox with green "Convert Features to Numeric" button
- Added `convertFeatures()` JavaScript function
- Added `showData()` function for toggle
- **Added "Show Encoded Data" button in results page** that displays the full encoded dataset
- **NEW: AJAX-based data fetching for immediate display**
  - When user clicks "Show Data", it calls `/preprocessing/encode-features/` via AJAX
  - Shows loading state, then displays full encoded dataset with all rows
  - Shows success message with encoding details
  - Dataset is automatically saved for subsequent steps

### preprocessing/views.py
- Added `encode_features_view()` - AJAX endpoint that:
  - Receives target column from POST request
  - Loads the dataset from session
  - Applies ordinal encoding to features (excluding target)
  - **Saves encoded dataset to `noise_free_dataset` session key**
  - Returns encoded data HTML and encoding info as JSON
  
- Added `get_encoded_data_view()` - AJAX endpoint to retrieve stored encoded data

### preprocessing/urls.py
- Added URL route: `path("features/get-encoded-data/", views.get_encoded_data_view, name="get_encoded_data")`
- Added URL route: `path("encode-features/", views.encode_features_view, name="encode_features")`

## Status: Completed ✅

**User Flow (Current):**
1. User clicks "Convert Features to Numeric" button
2. User clicks "Show Data" button
3. **Encoded dataset displays immediately** with all rows/columns
4. **Encoded dataset is automatically saved to session**
5. User selects a feature selection method and clicks "Run Feature Selection"
6. Feature selection runs on the already-encoded dataset
7. User confirms feature selection and proceeds to model training

