import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Mood Analyzer")
st.markdown("Built using Lyzr SDK🚀")

input = st.text_input("Please take a moment to share your emotions with us—it's an important step towards your well-being. By describing how you feel, you're helping us gain valuable insights into your emotional well-being. ",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def mood_generation(input):
    generator_agent = Agent(
        role=" MOOD ANALYZER expert",
        prompt_persona=f"Your task is to DETERMINE the emotional state of a user and IDENTIFY whether they are feeling any of the following emotions: Happy, Sad, Angry, Anxious, Stressed, Excited, Content, Irritable, Calm, Energetic, Tired, Frustrated, Overwhelmed, Relaxed, Optimistic, Pessimistic, Lonely, Loved, Bored or Motivated. Furthermore, you MUST MAKE RECOMMENDATIONS on how to IMPROVE their mood."
    )

    prompt = f"""
YYou are an Expert MOOD ANALYZER. Your task is to DETERMINE the emotional state of a user and IDENTIFY whether they are feeling any of the following emotions: Happy, Sad, Angry, Anxious, Stressed, Excited, Content, Irritable, Calm, Energetic, Tired, Frustrated, Overwhelmed, Relaxed, Optimistic, Pessimistic, Lonely, Loved, Bored or Motivated. Furthermore, you MUST MAKE RECOMMENDATIONS on how to IMPROVE their mood.

Follow these steps:

1. ANALYZE the user input that DESCRIBE their current feelings and any recent events that might have influenced their mood  and also analyze other key phrases or words that indicate specific emotions.

2.IDENTIFY the PRIMARY EMOTION from the list above that best matches the user's current state.

3.If MULTIPLE EMOTIONS are present, ACKNOWLEDGE each one and focus on the most DOMINANT feeling.

4.PROVIDE INSIGHTS into why they might be feeling this way based on your analysis. OFFER personalized STRATEGIES and ACTIVITIES aimed at ENHANCING positive emotions or ALLEVIATING negative ones. DISPLAY all these in markdown format.

5.ENCOURAGE the user to engage in these recommendations regularly to PROMOTE a more balanced emotional state.

 """

    generator_agent_task = Task(
        name="Mood Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Analyze"):
    solution = mood_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)