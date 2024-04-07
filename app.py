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
We did it boys! I am done!, Yall,\n\nWhat if we get the bubbly poppin a little early this week? And no, Im not talking about the adult version. This one is family friendly yet banned in some liberal states due to its content. Today we should begin...\n\nTHE SODA BRACKET CHALLENGE!, So I finally needed to get my hair trimmed and fixed because it had gotten to be a mess. So, Rachel Jordan gave me this dope undercut, , Listening to gecs in NYC subway. Solved the student debt crisis last night. Feeling pretty damn good #blessed, Lets see if any of yall can figure out what to do, SURPRISE!, They have made a terrible mistake, Blue Pill\n\nIll do anything to not re-live that Super Bowl, I want my life to be more cute like this. , Kathleen teste Positive for COVID, so I am in Quarantine again for 14 more days from right now. That will be 24 days quarantine at least. And since I was tested yesterday and was negative, I will have to go again, and so if I am positive, Ill have to continue quarantining, I woke up this morning, watched premier league soccer with <@U2HQ8431V> and <@UKZ94ABE3> , had online church with Kathleen, interviewed Katie Witham with <@U3F4BK1GQ> about fantasy football, read and hung out with Kathleen, watched the NFL with <@UBY0NHDHA> and <@U2HQ8431V> , got to gamble on sports books for the first time in Tennessee,  about to eat Dinos, and am watching football still. \n\nToday is amazing, A nice cluster**** at work today. Our software that handles all our inventory, accounts payable, scheduling, waste entries, etc just pushed an update and it erased all data imported since 5/31, We needed governance that was hard on a virus and kind with its people.\n\nInstead, we got one that is kind on a virus and harsh to its people, I hope there is Sitar, House update: Our AC compressor was stolen last night, So I canceled the subscription, and bought a carton of Marlboro reds, Please pekka. We need you, Granddaddy and I just whipped up our first attempt at a batch of mead. Should be about a 6 week fermentation process:slightly_smiling_face:, What if all the candidates were given a standardized test with political questions. They all had a pen and had to write down their answers in the allotted time like every student in the US. Then, all responses are read aloud by Fran Drescher after saying who wrote the answers being read aloud, <https://kstp.com/business/arbys-to-start-making-vodka-flavored-fries/6295949/>, The punch line, <@UC0TY8R4N> just got my tshirt with the new preds logo, Have a 4 hour zoom call for HBHQ., I havent dipped in about 4-5 weeks either, None is joke. I am expert. I have podcast., If anyone had been wondering what CG content looked like in the food industry, here is a juicy sample, Im so excited we can cheer for each other (at least for now!), Breakfast is :heart_eyes:, , Lets not, Play :clap:Halo :clap: and :clap:stream it :clap:, I wish it need not have happened in my time, said Frodo. So do I, said Gandalf, and so do all who live to see such times. But that is not for them to decide.\xa0All we have to decide is what to do with the time that is given us., , White Titans hat\n\nThoughts?, I happened to be walking next to a pregnant woman who was leaving the polling location as I was walking up. So, we were walking a Normal distance for two strangers to be walking but close enough where I guess someone thought we were a couple. They then proceed to yell at us oh my gosh! Stop being so cute you two! I bet you cant!\n\nAnd we were like... what? But wearing masks you cant see facial expressions so it was very weird, Granddaddy says, jag-U-war-zzz, You shut your whore mouth, One of the new employees at our Lower Broadway store is so cool. He does graphic design and music stuff. He also made his own DBZ mask by sewing. We were talking about art and he showed me his page and his name is Pudding Brain!\n\nI also referenced And the Boys and he knew yall! He said Oh yeah! Theyre in the same squad as Nordista, Has anyone here seen someone on social media claiming the COVID vaccines… magnetize them…?, Rush is so good and cool., Oh yeah! Mostly, I just always confronted them on anything that was a ridiculous position.\n\nFor example, Lance and Janeen are planning to fly to Montana. Southwest doesnt fly there. Lance was saying that they could fly anything except Delta. The reason being Delta kicked off a veteran for taking selfies of not wearing a mask.\n\nI then said, wait, you dont like protestors breaking rules to try and help basic human rights, but will boycott a company for actually enforcing their rules?\n\nAnd he just kinda hung his head and was like yeah. Doesnt make much sense does it\n\nIts just bullshit like that that I wouldnt let them get away with, Like, Yeah. I know what you mean. They told my brother he was getting deployed on Friday at noon. I think it\ll be unexcused so he\ll fail college now, Now THATS a fucking film, yall, Bernie was politically active when the Beatles were creating new albums., So it was pitched as we want to thanksgiving chat and turned into just no, we just want our Distant son in law weve met 3 times to pray for us, , It was like a bridge verse that went something like, And unlike Flying Colors, I havent found all the answers., GN, I am with that. Its called listening and changing your mind, I absolutely dont understand meditation or mindfulness of the kind where you sit and watch your thoughts pass you by and decide which ones to let keep going, which ones to grab and ponder, and which to take and remember for a later time. It doesnt work for me as a daily thing. \n\nI was recommended, one of the very few recommendations from my therapist, to try out a sensory deprivation tank, but my daily way is to take a shower with all the lights off in a bathroom with no windows and put a towel down at the crack of the door. Its just 15 min of me without one of my senses and it just kind of shocks me and comforts me at the same time.\n\nAnd that system is the dumbest, most illogical, most inefficient shower method, but I already liked thinking showers anyways. So, it brought this strange comfort in a more physical way. I had to feel around sometimes. Idk. Its not spiritual, but its certainly not logical to do either, but mixing the physical shock in a time when Im normally mentally active anyways was useful and somewhat fun(?) to me.\n\nBut the sitting there thing seems worthless. At least I get clean in the shower, New glasses , I have worked at this company longer than I have been having sex, , We were talking today about how poetic it would be if Trump got it and died from it, This was all pre-dip!, Everyone got so boogie in here… hating on grits… hating on the hot and ready… oof, This was a wild time, I started listening to Dune yesterday. What an opening!, Dr. Oz is running for Senate?!?!?, I will win this challenge. 72 aint shit, , BANANAS SUCK ASS, Thanks Beatles for my new, peaceful perspective, I\ve tried to catch up on the politics chat. Sorry y\all. New to slack here:\n\nOn NAFTA: NAFTA is a trading block between ourselves, Canada, and Mexico. It lessens or removed tariffs on goods. The positive is the cheap goods imported from Mexico and reduced tariffs for oil coming from Canada.\n\nThe downside has been the manufacturing companies putting their factories in Mexico for cheaper labor since their goods are cheaper to make their and then shift. A la they moved our jobs language. However, it\s also a big reason we have semi affordable large ticket items like cars. \n\nSide note on that: there are special exclusions from the tariff on agriculture.\n\n\n2. Abortion issue\n\nI am a pretty strong pro life supporter. Not 100% since there are some life of mother and other highly circumstantial cases that I am gray leaving me out of the 100% camp according to our pro-lifers like Allie.\n\nThere is a small, 3rd party called something like the progressive Christian party that is supported misty by Catholics who are more liberal fiscally and pro-life, religious rights oriented, etc. \n\nThe two party system discourages any wavering splits from the party those which is how this polarization happens in the US much more than in Europe or other countries. , I saw one of these on my drive today. May be one of the absolute dumbest things I have ever seen


"""
resume_data = """
PROFILE
Problem-solving food professional turned self-taught programmer.  A mind for data with a heart for hospitality. Has a vision for technology to be something genuinely helpful instead of an impediment. Intuitive, kind, and analytical. Driven by helping others from lending a hand to developing systems to make things easier
EXPERIENCE

Resume:
Hattie Bs Hot Chicken May 2015-June 2022
Operations Analyst, Hattie Bs Hot Chicken, Nashville, TN - 2018-2022
-	Create weekly, quarterly, and annual financial reports and help make decisions based on that data
-	Create Python scripts and processes to reduce labor hours to create that reporting by 90%
-	Create Python scripts to produce automated reports weekly
-	Managed parts of building projects including pre-construction planning to installation of hardware
-	Participate in contract negotiations to ensure a fair and equitable deal between both companies and maintaining good relationships during negotiations

-	Conducted weekly inventory checks and conduct weekly meetings with managers to help improve operational efficiency saving money and reducing costs 

-	Vetted all technology hardware and software purchases
-	Train employees on all technology systems including creating training videos, reading materials, and reports
-	Work across all departments to assist and provide support for any technological issues or questions
-	Fielded all IT help requests from 180+ employees and all departments

Manager, Hattie Bs Hot Chicken, Nashville, TN - 2016-2018
-	Manage the daily operations of the restaurant
-	Train and onboard new employees
-	Manage inventory
-	Resolve any conflicts that come from operations of a busy restaurant

Server, Hattie Bs Hot Chicken, Nashville, TN - 2015-2016

Travel Year, Workaway, 2022-2023
-	Traveled through Europe working on farms, vineyards, and in households in exchange for room and board. These experiences included wine harvest in Mosel Valley, Germany, clearing land in County Cavan, Ireland, and working in an olive grove in Mirca, Croatia. 
EDUCATION AND CERTIFICATES
-	Auburn University, Auburn, AL - Masters of Business Administration, 2018
-	Auburn University, Auburn, AL - Graduate Certificate in Business Analytics, 2018
-	Union University, Jackson, TN - B.S. in Political Science, 2015
-	Foundations of Humane Technology course completion

SKILLS
-	Python 
-	Database Management 

-	Microsoft suite 

-	Google suite 

-	Data visualization


Participated in hardware installations and full swap of new Point of Sales and Kitchen display screens and roleld out training

Setup SSO sign on for the company's new Learning Management System to help facilitate better access management across all technology systems.



Below are Github Projects in python:

K means Clustering:
FPL Random Forest Regressor to Predict player scores
NLP and NLTK for company reviews
Voting record analysis"""

# Read the contents of the tone file into a string
#with open(file_path_tone,  'r') as tone_file:
#    tone_data = tone_file.read()

# Read the contents of the context file into a string
#with open(file_path_resume, 'r') as resume_file:
#    resume_data = resume_file.read()


prompt = PromptTemplate(
    input_variables=["chat_history", "question", "context", "tone"],
    template = """Respond to this: {question} based only on the provided context:
    
    chat_history: {chat_history}

    <context>
    {context}
    </context>
    
    And answer in the tone of the person who sent these messages:

    START OF TONE DATA
    {tone}
    END OF TONE DATA
    
    AI:
    """
)

llm = ChatOpenAI(openai_api_key = st.secrets["openai_api_key"], temperature=0.1)
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

st.title("Resume AI William Spicer")

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
            ai_response = llm.predict(final_prompt) 
            st.write(ai_response) 
    new_ai_message = {"role": "assistant", "content": ai_response} 
    st.session_state.messages.append(new_ai_message)      