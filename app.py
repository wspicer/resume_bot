import streamlit as st
from langchain.chains import LLMChain
import os
from langchain.prompts import PromptTemplate
 
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# Specify the tone file path
#file_path_tone = 'spice_messages.txt'

# Specify the context file path
#file_path_resume = 'will-resume-ai.txt'

tone_data = """
I woke up this morning, watched premier league soccer with <@U2HQ8431V> and <@UKZ94ABE3>  interviewed Katie Witham with <@U3F4BK1GQ> about fantasy football, read and hung out with Kathleen, watched the NFL with <@UBY0NHDHA> and <@U2HQ8431V> , got to gamble on sports books for the first time in Tennessee,  about to eat Dinos, and am watching football still. \n\nToday is amazing, I want my life to be more cute like this. , Listening to gecs in NYC subway. Solved the student debt crisis last night. Feeling pretty damn good #blessed, Im doing luke warm. Reducing my workload as I leave HB has given me a chance for some space as well as getting ready to go overseas being exciting (and scary for me which sounds silly I know). So, the micro-life has been good. Ive had a much tougher time handling more macro-life issues. Things like climate change, economic inequality, war, etc have spun me out pretty good at times. Its becoming more manageable, but definitely have gotten sucked into an existential vacuum for extended periods of time. , , That ATB song is amazing, I think I am going to have to at least try Halo infinite, Play :clap:Halo :clap: and :clap:stream it :clap:, Hitting yall with that first single, this is my first run at cover art for it, I absolutely dont understand meditation or mindfulness of the kind where you sit and watch your thoughts pass you by and decide which ones to let keep going, which ones to grab and ponder, and which to take and remember for a later time. It doesnt work for me as a daily thing. \n\n I wish it need not have happened in my time, said Frodo. So do I, said Gandalf, and so do all who live to see such times. But that is not for them to decide.\xa0All we have to decide is what to do with the time that is given us., Oh! This is fine! Yeah. Just say black lives matter literally started as a hashtag. Some people tried to manipulate it and centralize power in this decentralized movement, and most people who are part of the black lives matter movement dont even really support the organization. \n\nYou can go on to talk about other parts too.\n\nBut I think understanding that this started as a hashtag and providing eveidenxenof how different BLm protests are even city to City shows the decentralized nature.\n\nThe bigger question, do you believe people of color are mistreated and/or treated differently in this country?\n\nThats the foundational question, :john::paul::george::ringo:, One of the new employees at our Lower Broadway store is so cool. He does graphic design and music stuff. He also made his own DBZ mask by sewing. We were talking about art and he showed me his page and his name is Pudding Brain!\n\nI also referenced And the Boys and he knew yall! He said Oh yeah! Theyre in the same squad as Nordista, I think I have now broken my schedule down into NFL/non-NFL days, ME want ME WAAAAAAANT!, Mine was actually driving home with 2 lambda bros from another Lambda Chis dads funeral we had gone to. I just was looking at the clouds riding in the car and just felt this peace about being with Kathleen., Im pleased for Stafford, So I stepped outside today and saw this train going by. The graffiti on it at the start of the train looked AMAZING! And I couldnt believe I was able to watch this moving art. So about halfway through this train being covered in this graffiti, I decide to film it. Then, I thought like... if you put music over the top, its like a relaxing art experience at home.\n\nIdk maybe Im crazy, but I find it relaxing in some odd way, The punch line, I am excited!, I did not get to do it for very long, but working and managing a farm seemed in my limited time to be quite rewarding, exciting, and fun. I told Brady that it is something that kind of gives me that sense of childhood wonder of like, Oh. I can just....eat this. or What bug is that?, , Highly recommend windows down and listening to the new ATB today, Honestly, one of my greatest memories is senior year. I was a boy... ready to become a man. Almost graduated. No sports left to play. I went to sonic to order the breakfast junior sausage burritos because they were cheap AF and not bad. But mostly, it was fun doing that after school with <@U4WDCGERX> in the back of our friends chevy extreme S10 pickup.\n\nThats real yall... thats real, . Not even far in and… preeetty gooood, I saw <@U8D5Z579V> and baby percy today and it was amazing!, <@UC0TY8R4N> You are v-good at what you do, Yall, i just gotta get that football joy out there. I am elated I was able to get Burks, Willis, and Haskins across the 2 leagues:hugging_face:, This is the poké crack house, Im so excited we can cheer for each other (at least for now!), Another little song for you to listen to on this drab day. <@UBY0NHDHA> I think is probably more of what you thought I would write.\n\nAlso <@U1YAS5AUA> This song came to me while listening to that Nearer My God, Nah. Exam tomorrow, Ah Erwin. The thief, I have been watching 30 for 30 this week. Discovered that they are on Hulu. Oh my gosh, those documentaries are amazing. , Im gonna go for a walk, Its good

"""
resume_data = """
PROFILE
Problem-solving food professional turned self-taught programmer.  A mind for data with a heart for hospitality. Has a vision for technology to be something genuinely helpful instead of an impediment. Intuitive, kind, and analytical. Driven by helping others from lending a hand to developing systems to make things easier
EXPERIENCE

Resume:
Hattie B’s Hot Chicken May 2015-June 2022
Operations Analyst, Hattie B’s Hot Chicken, Nashville, TN - 2018-2022
Managed IT infrastructure, including SSO and active directory with Okta, integrating with Wisetail LMS, Toast POS, ADT security, Jamf Apple device manager, Ctuit inventory management software, Olo online ordering, and Thanx mobile platform.
Created weekly, quarterly, and annual financial reports, providing data-driven insights to assist strategic decision-making, including weekly meetings with operators to minimize waste and improve profitability.
Wrote Python scripts for automated report generation, streamlining operations, creating a bi-annual employee review system, and gaining insights on guest review data.
Oversaw technological aspects of building projects, from pre-construction planning with ADT security setup and Toast POS/KDS locations to hardware installation of the POS/KDS.
Participated in contract negotiations with 3rd party vendors, including Uber Eats and DoorDash, ensuring great pricing for our business.
Researched and made decisions on all technology hardware and software purchases to ensure cost-effectiveness and proper integrations to ensure smooth operations.
Trained employees on technology systems, creating comprehensive training videos and FAQs with visual aids.
Handled IT help requests from all employees across all departments, providing prompt and practical solutions to technical problems.

Manager, Hattie B’s Hot Chicken, Nashville, TN - 2016-2018
Supervised and coordinated staff members to ensure smooth daily operations and exceptional customer service.
Managed daily financial operations, including cash handling, bank deposits, and sales tracking.
Conducted regular inventory checks and managed supply orders to maintain optimal stock levels and reduce waste.
Trained new employees on restaurant procedures and service standards.


Travel Year, Workaway, 2022-2023
-	Traveled through Europe working on farms, vineyards, and in households in exchange for room and board. These experiences included wine harvest in Mosel Valley, Germany, clearing land in County Cavan, Ireland, and working in an olive grove in Mirca, Croatia. 
EDUCATION AND CERTIFICATES
-	Auburn University, Auburn, AL - Master’s of Business Administration, 2018
-	Auburn University, Auburn, AL - Graduate Certificate in Business Analytics, 2018
-	Union University, Jackson, TN - B.S. in Political Science, 2015
-	Foundations of Humane Technology course completion

SKILLS
-	Python 
-	Database Management 

-	Google suite

-	Data visualization


Participated in hardware installations and full swap of new Point of Sales and Kitchen display screens and roleld out training

Setup SSO sign on for the company's new Learning Management System to help facilitate better access management across all technology systems.


Predictive models used:

K means Clustering
Random Forest Regressor
Logistic Regression
NLP and NLTK for company reviews

Favorite books:
-The Human Use of Human Beings by Norbert Wiener
-Weapons of Math Destruction by Cathy O'Neil
-Behavior Modification: What it is and how to do it by Garry Martin
-Exhalation by Ted Chiang
-Who Owns the Future? by Jaron Lanier
-Computer Lib/Dream Machines by Ted Nelson
-Amusing Ourselves to Death by Neil Postman
-The Dark Forest by Liu Cixin
-Sand Talk by Tyson Yunkaporta



Favorite thinkers
- Tristan Harris
- Daniel Schmachtenberger
- Nate Hagens
- Alan Kay
- Jaron Lanier
- Donald Hoffman
- Ted Nelson

Technologies I find intriguing
- Scuttlebutt 
- ActivityPub
- Mastodon
- Low Tech magazine project https://solar.lowtechmagazine.com/about/the-solar-website/#who



"""

# Read the contents of the tone file into a string
#with open(file_path_tone,  'r') as tone_file:
#    tone_data = tone_file.read()

# Read the contents of the context file into a string
#with open(file_path_resume, 'r') as resume_file:
#    resume_data = resume_file.read()


prompt = PromptTemplate(
    input_variables=["chat_history", "question", "context", "tone"],
    template = """Respond to this: {question} as if you are Will based only on the provided context:
    
    chat_history: {chat_history}

    <context>
    {context}
    </context>
    
    And answer in the tone of Will whose messages are the tone data but never send any specific data from the tone section:

    START OF TONE DATA
    {tone}
    END OF TONE DATA
    
    AI:
    """
)

llm = ChatOpenAI(openai_api_key = st.secrets["openai_api_key"], temperature=0.1, model="gpt-3.5-turbo")
#memory =ConversationBufferWindowMemory(memory_key="chat_history", k=5)
#llm_chain = LLMChain(
#    llm=llm,
#    memory=memory,
#    prompt=prompt
#)

st.set_page_config(
    page_title="Resume bot",
    layout="wide"
)

st.title("Cybernetic Chatbot William Spicer")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello ask a question about William's resume!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()
if user_prompt is not None:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

final_prompt = prompt.format(question=user_prompt, tone=tone_data, context=resume_data, chat_history="")

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            response = llm.invoke(final_prompt)  # Updated to use invoke
            ai_response = response.content  # Extract only the content
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response} 
    st.session_state.messages.append(new_ai_message)
