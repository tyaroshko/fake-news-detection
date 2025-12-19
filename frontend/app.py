import asyncio
import time
from typing import Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="News Classification App",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = "http://localhost:8000"


class NewsClassifierClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def get_available_models(self) -> List[Dict]:
        """Get list of available classification models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/models")
            response.raise_for_status()
            data = response.json()
            return data["models"]
        except Exception as e:
            st.error(f"Failed to fetch models: {e}")
            return []

    async def classify_text(
        self, text: str, model_name: str, file_name: Optional[str] = None
    ) -> Optional[Dict]:
        """Classify text content."""
        try:
            payload = {"text": text, "model_name": model_name, "file_name": file_name}
            response = await self.client.post(
                f"{self.base_url}/api/classify/text", json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            return None

    async def classify_file(
        self, file_content: bytes, filename: str, model_name: str
    ) -> Optional[Dict]:
        """Classify file content."""
        try:
            files = {"file": (filename, file_content, "text/plain")}
            data = {"model_name": model_name}
            response = await self.client.post(
                f"{self.base_url}/api/classify/file", files=files, data=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"File classification failed: {e}")
            return None

    async def get_classification_history(self, limit: int = 50) -> List[Dict]:
        """Get classification history."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/results?limit={limit}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")
            return []


# Initialize client
client = NewsClassifierClient(API_BASE_URL)


def normalize_classification_result(result: Dict) -> tuple[str, float]:
    """Normalize classification result to extract prediction and confidence from any response format."""
    if "classification" in result:
        # LLM response format
        return result["classification"]["result"], result["classification"][
            "confidence"
        ]
    else:
        # BERT response format
        return result["prediction"], result["confidence"]


def main():
    st.title("📰 News Classification App")
    st.markdown("Classify news articles as real or fake using advanced AI models")

    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        if "available_models" not in st.session_state:
            with st.spinner("Loading models..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                st.session_state.available_models = loop.run_until_complete(
                    client.get_available_models()
                )

        if st.session_state.available_models:
            model_options = [
                f"{model['name']} ({model['type'].upper()})"
                for model in st.session_state.available_models
            ]
            model_display_map = {
                opt: model["name"]
                for opt, model in zip(model_options, st.session_state.available_models)
            }

            selected_model_display = st.selectbox(
                "Select Classification Model",
                options=model_options,
                help="Choose from BERT-like models or Large Language Models",
            )
            selected_model = model_display_map[selected_model_display]

            # Show model info
            model_info = next(
                (
                    m
                    for m in st.session_state.available_models
                    if m["name"] == selected_model
                ),
                None,
            )
            if model_info:
                st.info(f"**{model_info['name']}**\n\n{model_info['description']}")
        else:
            st.error("Failed to load models. Please check if the backend is running.")
            selected_model = None

        st.divider()

        # API Status
        if st.button("🔄 Check API Status"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    client.client.get(f"{API_BASE_URL}/health")
                )
                if response.status_code == 200:
                    st.success("✅ API is running")
                else:
                    st.error("❌ API is not responding")
            except:
                st.error("❌ Cannot connect to API")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(
        ["📝 Text Classification", "📁 File Upload", "📊 History"]
    )

    # Tab 1: Text Classification
    with tab1:
        st.header("Classify Text")

        text_input = st.text_area(
            "Enter news article text:",
            height=200,
            placeholder="Paste your news article text here...",
            help="Enter the news article content you want to classify",
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            classify_text_btn = st.button(
                "🔍 Classify Text",
                type="primary",
                disabled=not text_input.strip() or not selected_model,
                use_container_width=True,
            )

        with col2:
            clear_text_btn = st.button("🗑️ Clear", use_container_width=True)

        # Classification results for text
        if "text_result" not in st.session_state:
            st.session_state.text_result = None

        if classify_text_btn and text_input.strip() and selected_model:
            with st.spinner("Analyzing text..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    client.classify_text(text_input, selected_model)
                )
                st.session_state.text_result = result

        if clear_text_btn:
            st.session_state.text_result = None
            text_input = ""

        # Display text classification result
        if st.session_state.text_result:
            display_classification_result(st.session_state.text_result)

    # Tab 2: File Upload
    with tab2:
        st.header("Classify File")

        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=["txt", "md", "csv"],
            help="Upload a text file containing news content",
        )

        if uploaded_file and selected_model:
            st.success(f"File uploaded: {uploaded_file.name}")

            if st.button("🔍 Classify File", type="primary"):
                file_content = uploaded_file.getvalue()

                with st.spinner("Analyzing file content..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        client.classify_file(
                            file_content, uploaded_file.name, selected_model
                        )
                    )

                    if result:
                        display_classification_result(result)

    # Tab 3: Classification History
    with tab3:
        st.header("Classification History")

        if st.button("🔄 Refresh History"):
            with st.spinner("Loading history..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                history = loop.run_until_complete(
                    client.get_classification_history(100)
                )
                st.session_state.history = history

        if "history" not in st.session_state:
            st.session_state.history = []

        if st.session_state.history:
            # Convert to DataFrame for better display
            # Normalize the data structure to handle both BERT and LLM responses
            normalized_data = []
            for item in st.session_state.history:
                normalized_item = {
                    "id": item["id"],
                    "text_content": item["text_content"],
                    "file_name": item.get("file_name"),
                    "model_used": item["model_used"],
                    "created_at": item["created_at"],
                }

                # Handle different response formats
                if "classification" in item:
                    # LLM response format
                    normalized_item["prediction"] = item["classification"]["result"]
                    normalized_item["confidence"] = item["classification"]["confidence"]
                else:
                    # BERT response format
                    normalized_item["prediction"] = item["prediction"]
                    normalized_item["confidence"] = item["confidence"]

                normalized_data.append(normalized_item)

            df = pd.DataFrame(normalized_data)

            # Format the dataframe
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df["confidence"] = df["confidence"].round(3)
            df["text_preview"] = df["text_content"].str[:100] + "..."

            # Reorder columns
            display_df = df[
                [
                    "created_at",
                    "model_used",
                    "prediction",
                    "confidence",
                    "file_name",
                    "text_preview",
                ]
            ]

            st.dataframe(
                display_df,
                column_config={
                    "created_at": st.column_config.TextColumn("Date", width="medium"),
                    "model_used": st.column_config.TextColumn("Model", width="medium"),
                    "prediction": st.column_config.TextColumn(
                        "Prediction", width="small"
                    ),
                    "confidence": st.column_config.NumberColumn(
                        "Confidence", format="%.3f", width="small"
                    ),
                    "file_name": st.column_config.TextColumn("File", width="medium"),
                    "text_preview": st.column_config.TextColumn(
                        "Content Preview", width="large"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_classifications = len(df)
                st.metric("Total Classifications", total_classifications)

            with col2:
                real_count = len(df[df["prediction"].str.lower() == "real"])
                st.metric("Real News", real_count)

            with col3:
                fake_count = len(df[df["prediction"].str.lower() == "fake"])
                st.metric("Fake News", fake_count)

        else:
            st.info(
                "No classification history available. Start classifying some content!"
            )


def display_classification_result(result: Dict):
    """Display classification result in a nice format."""
    st.success("✅ Classification Complete!")

    col1, col2, col3 = st.columns(3)

    # Normalize the result data
    prediction, confidence = normalize_classification_result(result)
    prediction = prediction.upper()

    with col1:
        if prediction == "REAL":
            st.metric("Prediction", "🟢 REAL NEWS")
        elif prediction == "FAKE":
            st.metric("Prediction", "🔴 FAKE NEWS")
        else:
            st.metric("Prediction", f"❓ {prediction}")

    with col2:
        st.metric("Confidence", f"{confidence:.3f}")

    with col3:
        model_used = result["model_used"]
        st.metric("Model Used", model_used)

    # Additional details
    with st.expander("📋 Details"):
        st.write(f"**ID:** {result['id']}")
        st.write(f"**File:** {result.get('file_name', 'N/A')}")
        st.write(f"**Timestamp:** {result['created_at']}")

        # Show confidence gauge
        if 0 <= confidence <= 1:
            st.progress(confidence)
            if confidence > 0.8:
                st.write("🔥 High confidence prediction")
            elif confidence > 0.6:
                st.write("⚠️ Moderate confidence prediction")
            else:
                st.write("🤔 Low confidence prediction")


if __name__ == "__main__":
    main()
