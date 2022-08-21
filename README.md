# Manual
### Hello! **We're growing sprouts, team sprout.** 👋
We use Microsoft TEAMS, Azure and Power Apps to make **'sprout'**. For more information, see [Power Apps for developers](https://docs.microsoft.com/powerapps/#pivot=home&panel=developer). 


# 🌱 Introduction
With the increase of remote work and multiple chat platforms, communication challenges only seem to increase, which can become a roadblock to productive collaboration. To help individuals keep up to date with any progress, we provide 🌱 **“Chat Q&A and Summarization Program”.** 🌱


# 🌱 Function
**Summarization Program** : Users can capture specific range of dialogues using simple command. And the NLPCloud API gets these dialogue contexts from Microsoft Teams to summarize for users using conversation summarization fine-tuned BART model.

**Chat QA** : The app serves as a Queation and Answering chatbot that gives work-related answers based in previous conversations. It is based on RoBERTa with squad2 dataset fine-tuned in HuggingFace API. With this, people can making the overall process more efficient.


# 🌱 Development Environment
- Microsoft’s Power Automate
- Azure Cosmos DB
- Power Apps
- PyTorch and HuggingFace, NLPCLoud API for ML
- AWS Lambda, EC2, API gateway
We use Microsoft’s Power Automate, Azure, and Power apps to make **'sprout'**. With these low-/no-code way, we can develop worth app despite of short time.


# 🌱 Setting

### **Install and run applications**

1. You can run our app with any device such as phone, or tablet. If you are an Android user, download **'Power apps'** through the Google Store and if you are an Apple user, download **'Power apps'** through the App Store.

![KakaoTalk_Photo_2022-08-21-10-19-26](https://user-images.githubusercontent.com/76519535/185772159-5248fd6d-8b14-4903-b3b0-a90e1e08e785.jpeg)

2. Please download and run the app. After that, Please log in to the app with your Microsoft account.

![KakaoTalk_Photo_2022-08-21-10-33-05](https://user-images.githubusercontent.com/76519535/185772171-4ab6da43-b738-45af-8a12-844f3291a2ef.jpeg)

3. After running the application, press **‘all apps’** button in the lower bar. You can see the app we built. Press **‘JA Sprout’** to start!

![KakaoTalk_Photo_2022-08-21-10-33-02](https://user-images.githubusercontent.com/76519535/185772175-d43d60f7-97f0-492e-a005-faa24f8df539.jpeg)
![KakaoTalk_Photo_2022-08-21-10-32-59](https://user-images.githubusercontent.com/76519535/185772178-8477b37b-ee18-46ba-b261-9bcdeca0b0cb.jpeg)



# 🌱 To Use app

‼️ Before use app, You need access to our onedrive, because we use excel to store data received from api. For this, you should use the account of the user who belongs to our tenant. Or, you can access our one drive with the access link. 


‼️ Essential Excel File that uploaded in onedrive 
- summary/GENERAL1.xlsx (be used to get summarize context)
- /Query-Results-5.xlsx (be used to get channel of Teams)

1. If you look at the top left, you can see our two main functions: **'Answer a Question'** and **'Summarize'.** Click each button to move the window!
2. First, the **"Answer a Question"** window. The list on the left calls up the Teams channel you are currently accessing. When you click a channel, a QA bot that uses AI to analyze the channel history is launched.
    
    ![KakaoTalk_Photo_2022-08-21-10-35-20](https://user-images.githubusercontent.com/76519535/185772184-3402dfa3-165a-4bce-b4c9-3bc2e1c6a1d1.jpeg)
    
3. Next is the **'Summarize'** window. Similarly, on the left side, the Teams channel you are currently accessing is called up. When you select a channel, the list of conversations recorded on that channel appears briefly summarized.
    
   ![KakaoTalk_Photo_2022-08-21-10-35-26](https://user-images.githubusercontent.com/76519535/185772186-59fb302b-baaf-4565-8861-356a1e9d3cbd.jpeg)

