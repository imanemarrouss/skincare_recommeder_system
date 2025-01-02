

import torch
from torchvision import transforms
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import numpy as np
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cache model loading to avoid reloading the model
@st.cache_resource
def load_classification_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


@st.cache_resource
def load_skin_defect_model():
    return tf.keras.models.load_model(r"C:\Users\Dell\Downloads\my_model.h5")

# Cache data loading and processing for the new dataset

skin_defect_model = load_skin_defect_model()

classification_model = load_classification_model(r'C:\Users\Dell\Downloads\best_model (1).pth')


def preprocess_single_image(image_file):
    img = img_to_array(load_img(image_file, target_size=(224, 224)))
    img = img / 255.0  # Normalize pixel values

    # If your model expects 9 channels, you might want to create an input with 9 channels.
    # Here, we'll repeat the image channels 3 times to create a 9-channel image.
    img = np.repeat(img, 3, axis=-1)  # Repeat the 3 channels 3 times to create 9 channels
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Define class names for the skin defect model
defect_class_names = ["Acne", "Bags", "Redness"]

def predict_skin_defect(image_file):
    input_data = preprocess_single_image(image_file)
    predictions = skin_defect_model.predict(input_data)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return defect_class_names[class_idx], confidence

# Preprocess image
def preprocess_image(image):
    image = image.convert('RGB').resize((224, 224))
    img_array = np.transpose(np.array(image) / 255.0, (2, 0, 1))[None, ...]
    return torch.tensor(img_array, dtype=torch.float32)

# Predict skin type
def predict_skin_type(image):
    processed_image = preprocess_image(image)
    predictions = classification_model(processed_image)
    class_names = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
    return class_names[torch.argmax(predictions).item()]

# Cache data loading and processing
@st.cache_data
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path).dropna().drop_duplicates()
    data = data.rename(columns={'Reviewer': 'reviewer', 'Product': 'product', 'Stars': 'rating'})
    return data[['reviewer', 'product', 'rating']]

@st.cache_data
def load_and_prepare_updated_data(file_path):
    data = pd.read_csv(file_path).dropna().drop_duplicates()
    return data[['product_name', 'product_url', 'product_type', 'clean_ingreds', 'price', 'combined_text', 'defect']]

# Build collaborative filtering recommendation model
@st.cache_data
def build_recommendation_model(data):
    matrix_pivot = pd.pivot_table(data, values='rating', index='reviewer', columns='product').fillna(0)
    user_ratings_mean = np.mean(matrix_pivot.values, axis=1)
    user_rating = matrix_pivot.values - user_ratings_mean[:, None]

    U, sigma, Vt = svds(user_rating, k=50)
    all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean[:, None]
    return pd.DataFrame(all_user_predicted_ratings, index=matrix_pivot.index, columns=matrix_pivot.columns)


# Content-based recommendation function for the new dataset
def recommend_cosmetics_based_on_ingredients(df, ingredient_input=None, num_recommendations=10):
    # Vectorize ingredients
    vectorizer = TfidfVectorizer(stop_words='english')
    cosine_similarities = cosine_similarity(vectorizer.fit_transform(df['clean_ingreds']), vectorizer.transform([ingredient_input])).flatten()
    df['similarity'] = cosine_similarities
    
    # Sort products by similarity score
    df_sorted = df.sort_values(by='similarity', ascending=False)
    
    # Return the top recommendations
    return df_sorted.head(num_recommendations)


def recommend_cosmetics_based_on_skin_concern(df, skin_concern=None, ingredient_input=None, num_recommendations=10):
    # Filter products based on the skin concern
    if skin_concern:
        df_filtered = df[df['defect'].str.contains(skin_concern, case=False, na=False)]
    else:
        df_filtered = df
    
    # If no ingredients are provided, return recommendations based on skin concern
    if ingredient_input:
        # Vectorize ingredients
        vectorizer = TfidfVectorizer(stop_words='english')
        cosine_similarities = cosine_similarity(vectorizer.fit_transform(df_filtered['clean_ingreds']), vectorizer.transform([ingredient_input])).flatten()
        df_filtered['similarity'] = cosine_similarities
        
        # Sort products by similarity score
        df_sorted = df_filtered.sort_values(by='similarity', ascending=False)
    else:
        # If no ingredient input, just recommend based on the skin concern filter
        df_sorted = df_filtered
        
    # Return the top recommendations
    return df_sorted.head(num_recommendations)


# Content-based recommendation function
def recommend_cosmetics(df, filters, ingredient_input=None, num_recommendations=10):
    filtered_df = df.copy()
    if filters['label'] != 'All':
        filtered_df = filtered_df[filtered_df['Label'] == filters['label']]
    filtered_df = filtered_df[(filtered_df['Rank'] >= filters['rank'][0]) & (filtered_df['Rank'] <= filters['rank'][1])]
    if filters['brand'] != 'All':
        filtered_df = filtered_df[filtered_df['Brand'] == filters['brand']]
    filtered_df = filtered_df[(filtered_df['Price'] >= filters['price'][0]) & (filtered_df['Price'] <= filters['price'][1])]
    if ingredient_input:
        vectorizer = TfidfVectorizer(stop_words='english')
        cosine_similarities = cosine_similarity(vectorizer.fit_transform(df['Ingredients']), vectorizer.transform([ingredient_input])).flatten()
        filtered_df['similarity'] = cosine_similarities
        filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    return filtered_df.head(num_recommendations)

def hybrid_recommendations(user_id, df, preds_df, content_filters=None, ingredient_input=None, alpha=0.6):
        """
        Combines collaborative and content-based filtering for hybrid recommendations.
        """
        # Collaborative Filtering Scores
        collaborative_scores = preds_df.loc[user_id] if user_id in preds_df.index else preds_df.mean(axis=0)

        # Content-Based Scores
        content_based_scores = pd.Series(index=collaborative_scores.index, data=0.0)  # Default to zero
        if content_filters or ingredient_input:
            try:
                recommendations = recommend_cosmetics(df, content_filters, ingredient_input)
                content_based_scores = recommendations.set_index('Name')['score']
            except Exception as e:
                print(f"Error generating content-based scores: {str(e)}")  # Replace with logging if needed

        # Normalize Scores
        collaborative_scores = collaborative_scores / collaborative_scores.max() if collaborative_scores.max() > 0 else collaborative_scores
        content_based_scores = content_based_scores / content_based_scores.max() if content_based_scores.max() > 0 else content_based_scores

        # Hybrid Scores
        hybrid_scores = alpha * collaborative_scores + (1 - alpha) * content_based_scores.reindex(collaborative_scores.index, fill_value=0)

        # Rank Recommendations
        recommendations = pd.DataFrame({
            'Name': hybrid_scores.index,
            'score': hybrid_scores.values
        }).sort_values(by='score', ascending=False)

        return recommendations.head(10)


def main():

    st.title('Skincare Products Recommendation System')
    global preds_df
    # Upload image
    uploaded_image = st.file_uploader("Upload your skin image:", type=["jpg", "png"])
    predicted_skin_type = None
    predicted_skin_defect = None

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Predict skin type
        predicted_skin_type = predict_skin_type(Image.open(uploaded_image))
        st.write(f"**Predicted Skin Type:** {predicted_skin_type}")

        # Predict skin defect
        defect, confidence = predict_skin_defect(uploaded_image)
        predicted_skin_defect = defect
        st.write(f"**Detected Skin Defect:** {defect}")
        st.write(f"**Confidence:** {confidence:.2f}")

    # Recommendation type
    recommendation_type = st.radio(
        "Select Recommendation Method:",
        ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Filtering']
    )

    # if recommendation_type == 'Collaborative Filtering':
    #     user_id = st.text_input("Enter your user ID:")
    #     if user_id:
    #         user_data = data[data['reviewer'] == user_id]
    #         if user_data.empty:
    #             st.write(f"No data available for user ID: {user_id}")
    #         else:
    #             predictions = preds_df.loc[user_id].sort_values(ascending=False)
    #             st.write("Recommended products for you:")
    #             st.write(predictions.head(5))

    if recommendation_type == 'Collaborative Filtering':
        user_id = st.text_input("Enter your user ID:")
        if user_id:
            user_data = data[data['reviewer'] == user_id]
            if user_data.empty:
                st.write(f"No data available for user ID: {user_id}")
            else:
                predictions = preds_df.loc[user_id].sort_values(ascending=False)
                st.write("Recommended products for you:")
                st.write(predictions.head(5))

                # Calculate MAE and RMSE
                actual_ratings = user_data.set_index('product')['rating']
                predicted_ratings = predictions.reindex(actual_ratings.index, fill_value=0)
                mae = mean_absolute_error(actual_ratings, predicted_ratings)
                rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                # Calculate Precision@K and Recall@K
                k = 5
                actual_items = user_data['product'].tolist()
                predicted_items = predictions.index.tolist()
                precision = len(set(actual_items) & set(predicted_items[:k])) / k
                recall = len(set(actual_items) & set(predicted_items[:k])) / len(actual_items)
                st.write(f"**Precision@{k}:** {precision:.2f}")
                st.write(f"**Recall@{k}:** {recall:.2f}")

                
    # elif recommendation_type == 'Content-Based Filtering':
    #     filters = {
    #         'label': st.selectbox('Filter by label:', ['All'] + df_skin_type['Label'].unique().tolist()),
    #         'rank': st.slider('Rank range:', int(df_skin_type['Rank'].min()), int(df_skin_type['Rank'].max()), (1, 100)),
    #         'brand': st.selectbox('Filter by brand:', ['All'] + df_skin_type['Brand'].unique().tolist()),
    #         'price': st.slider('Price range:', float(df_skin_type['Price'].min()), float(df_skin_type['Price'].max()), (10.0, 100.0))
    #     }
    #     ingredient_input = st.text_area("Enter ingredients (optional):")
    #     if st.button("Find Recommendations"):
    #         recommendations = recommend_cosmetics(df_skin_type, filters, ingredient_input)
    #         st.write(recommendations)

    elif recommendation_type == 'Content-Based Filtering':
        filters = {
            'label': st.selectbox('Filter by label:', ['All'] + df_skin_type['Label'].unique().tolist()),
            'rank': st.slider('Rank range:', int(df_skin_type['Rank'].min()), int(df_skin_type['Rank'].max()), (1, 100)),
            'brand': st.selectbox('Filter by brand:', ['All'] + df_skin_type['Brand'].unique().tolist()),
            'price': st.slider('Price range:', float(df_skin_type['Price'].min()), float(df_skin_type['Price'].max()), (10.0, 100.0))
        }
        ingredient_input = st.text_area("Enter ingredients (optional):")
        if st.button("Find Recommendations"):
            recommendations = recommend_cosmetics(df_skin_type, filters, ingredient_input)
            st.write(recommendations)

            # Calculate Cosine Similarity
            if ingredient_input:
                vectorizer = TfidfVectorizer(stop_words='english')
                cosine_sim = cosine_similarity(vectorizer.fit_transform(df_skin_type['Ingredients']), vectorizer.transform([ingredient_input])).flatten()
                st.write(f"**Average Cosine Similarity:** {cosine_sim.mean():.2f}")

            # Calculate Precision@K and Recall@K (if ground truth is available)
            # Example: Assume ground truth is a list of relevant products
            ground_truth = ["Product A", "Product B"]  # Replace with actual ground truth
            k = 5
            predicted_items = recommendations['product_name'].tolist()
            precision = len(set(ground_truth) & set(predicted_items[:k])) / k
            recall = len(set(ground_truth) & set(predicted_items[:k])) / len(ground_truth)
            st.write(f"**Precision@{k}:** {precision:.2f}")
            st.write(f"**Recall@{k}:** {recall:.2f}")

        

    elif recommendation_type == 'Hybrid Filtering':

        #global preds_df  # Declare preds_df as a global variable
        st.title('Skincare Products Recommendation System')

        # Clean the price column
        df['price'] = df['price'].replace('£', '', regex=True).astype(float)

        # Collaborative Filtering Section
        st.header("Collaborative Filtering Recommendations")
        user_id = st.text_input("Enter your user ID:")
        
        if user_id:
            user_data = data[data['reviewer'] == user_id]
            if user_data.empty:
                st.write(f"No data available for user ID: {user_id}")
                st.write("Here are some popular products for you:")
                popular_items = data.groupby('product')['rating'].mean().sort_values(ascending=False).head(5)
                st.write(popular_items)
            else:
                predictions = preds_df.loc[user_id].sort_values(ascending=False)
                st.write("Recommended products for you:")
                st.write(predictions.head(5))

                # Rating Feedback
                st.write("**Rate the recommendations to help us improve:**")
                user_feedback = {}
                for product in predictions.head(5).index:
                    feedback = st.radio(f"How do you like {product}?", ["👍", "👎"], key=product)
                    if feedback == "👍":
                        user_feedback[product] = 1  # Positive feedback
                    else:
                        user_feedback[product] = 0  # Negative feedback

                # Save feedback to a file or database
                if st.button("Submit Feedback"):
                    import json
                    with open('user_feedback.json', 'w') as f:
                        json.dump(user_feedback, f)
                    st.write("Thank you for your feedback! Your preferences have been updated.")

                    # Retrain the model (optional)
                    for product, rating in user_feedback.items():
                        if user_id in data['reviewer'].values and product in data['product'].values:
                            data.loc[(data['reviewer'] == user_id) & (data['product'] == product), 'rating'] = rating
                    preds_df = build_recommendation_model(data)

                    # Regenerate recommendations
                    updated_recommendations = hybrid_recommendations(user_id, df, preds_df)
                    st.write("Updated recommendations:")
                    st.write(updated_recommendations.head(5))

                # Filtering Options
                price_range = st.slider("Filter by price range:", float(df['price'].min()), float(df['price'].max()), (10.0, 100.0))
                filtered_recommendations = predictions[predictions.index.isin(df[df['price'].between(price_range[0], price_range[1])]['product_name'])]
                st.write("Filtered recommendations:")
                st.write(filtered_recommendations)
        else:
            st.write("Please enter your user ID to get personalized recommendations.")




if __name__ == "__main__":
    # Load data and recommendation model only once
    data = load_and_prepare_data(r"C:\Users\Dell\Downloads\Female Daily Skincare Review Final.csv")
    preds_df = build_recommendation_model(data)
    df = load_and_prepare_updated_data(r"C:\Users\Dell\Downloads\updated_skin_care_products_with_random.csv")

    df_skin_type = pd.read_csv(r'C:\Users\Dell\Downloads\archive (2)\cosmetics.csv', delimiter=',')
    main()
