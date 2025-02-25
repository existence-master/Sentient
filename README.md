<div align="center">

![README Banner](./.github/assets/banner.png)
<h1>Your personal, private & interactive AI companion</h1>
  
<!-- Badges -->
<p>
  <a href="https://github.com/existence-master/Sentient/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/existence-master/Sentient" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/existence-master/Sentient" alt="last update" />
  </a>
  <a href="https://github.com/existence-master/Sentient/network/members">
    <img src="https://img.shields.io/github/forks/existence-master/Sentient" alt="forks" />
  </a>
  <a href="https://github.com/existence-master/Sentient/stargazers">
    <img src="https://img.shields.io/github/stars/existence-master/Sentient" alt="stars" />
  </a>
  <a href="https://github.com/existence-master/Sentient/issues/">
    <img src="https://img.shields.io/github/issues/existence-master/Sentient" alt="open issues" />
  </a>
</p>
   
<h4>
    <a href="https://www.youtube.com/watch?v=vIWgxLUlgR8/">View Demo</a>
  <span> · </span>
    <a href="https://beneficial-billboard-012.notion.site/Sentient-1916d7df083c809c929bf012b33594fc">Documentation</a>
  <span> · </span>
    <a href="https://github.com/existence-master/Sentient/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/existence-master/Sentient/issues/">Request Feature</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About The Project](#star2-about-the-project)
  * [Philosophy](#thought_balloon-philosophy)
  * [Screenshots](#camera-screenshots)
  * [Tech Stack](#space_invader-tech-stack)
  * [Features](#dart-features)
- [Roadmap](#compass-roadmap)
- [Getting Started](#toolbox-getting-started)
  * [Prerequisites](#bangbang-prerequisites-contributors)
  * [Installation](#gear-installation-users)
  * [Environment Variables](#-environment-variables-contributors)
  * [Run Locally](#running-run-locally-contributors)
- [Usage](#eyes-usage)
- [Contributing](#wave-contributing)
  * [Code of Conduct](#scroll-code-of-conduct)
- [FAQ](#grey_question-faq)
- [License](#warning-license)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)
- [Official Team](#heavy_check_mark-official-team)


<!-- About The Project -->
## :star2: About The Project

<!-- Philosophy -->
## :thought_balloon: Philosophy

We at [Existence](https://existence.technology) believe that AI won't simply die as a fad or remain limited to an assistant. Instead, it will evolve to be a true companion of humans and our aim with Sentient is to contribute to that future. Building a true companion requires excellent automation which in turn requires deep personalization. And if we want the AI to be completely context aware of the user, privacy is non-negotiable. Hence, our goal is to build a completely private, personal & interactive AI companion. And we want to build it in the open, not behind closed doors

<!-- Screenshots -->
### :camera: Screenshots

<div align="center"> 
  <img src="https://media.licdn.com/dms/image/v2/D4D22AQEHqTuOm0z9yw/feedshare-shrink_1280/B4DZTWSZA4HkAk-/0/1738761934906?e=1743033600&v=beta&t=Kteh7pToXInoAQ6WUCLmf9ABc8Yrod0meAcekNsmROc" alt="screenshot" />
  <p align="center">Agents in Sentient</p>
</div>
<div align="center"> 
  <img src="https://media.licdn.com/dms/image/v2/D4D22AQGzRW37TxbgDg/feedshare-shrink_2048_1536/B4DZS2U5tUG4Ao-/0/1738225720816?e=1743033600&v=beta&t=pZh3NCZCYDL3_IslFZAqxcK5nlQDLoZBYVWYUH-Gw2c" alt="screenshot" />
  <p align="center">Memories in Sentient</p>
</div>
<div align="center"> 
  <img src="https://github.com/user-attachments/assets/2e130ecf-b3ba-4074-8429-7be3fc241dae" alt="screenshot" />
  <p align="center">Local inference in Sentient</p>
</div>


<!-- TechStack -->
### :space_invader: Tech Stack

<details>
  <summary>Sentient Desktop App</summary>
  <ul>
    <li><a href="https://www.typescriptlang.org/">ElectronJS</a></li>
    <li><a href="https://nextjs.org/">Next.js</a></li>
    <li><a href="https://tailwindcss.com/">TailwindCSS</a></li>
    <li><a href="https://fastapi.tiangolo.com/">FastAPI</a></li>
    <li><a href="https://ollama.com/">Ollama</a></li>
    <li><a href="https://neo4j.com/">Neo4j</a></li>
    <li><a href="https://praw.readthedocs.io/en/stable/">PRAW Reddit API</a></li>
    <li><a href="https://pypi.org/project/linkedin-scraper/">LinkedIn Scraper</a></li>
    <li><a href="https://nitter-scraper.readthedocs.io/en/latest/">Nitter for X</a></li>
    <li><a href="https://developers.google.com/workspace">Google Workspace APIs</a></li>
    <li><a href="https://brave.com/search/api/">Brave Search API</a></li>
  </ul>
</details>

<!-- Features -->
### :dart: Features

- Completely Local and Private
- MBTI Personality Test
- LinkedIn, Reddit and X Integration for Personal Context
- Self-Managed Graph Memory - Sentient can learn about the user from their interactions with it. Memories are saved in a knowledge-graph.
- Agentic integrations for GSuite - Gmail, GCalendar, GDrive, GDocs, GSheets, GSlides support.
- Web Search capabilities - Sentient can search the web for additional context to answer queries
- Multi-chat functionality
- Auto-updates for the app


<!-- Roadmap -->
## :compass: Roadmap

* [x] Solving stability issues with model inference - improving structured outputs, better error handling, and more
* [ ] Dual Memory - Sentient will have a short-term memory and a long-term memory. The short-term memory will be stored in a relational DB and used to maintain reminders and other short-term information. Longer-term facts about the user will persist in the knowledge graph.
* [ ] Intent - Sentient will be able to perform actions autonomously based on triggers from the short-term memory.
* [ ] Tool-specific UI - for example, a tool that retrieves stock prices will also show a graph of the past trend of that particular ticker.
* [ ] Better Internet Search - Internet search will include images, citations, direct links to sources and more.
* [ ] More tools! - More tools will be added in the order of user requests. Current requests - Notion, LinkedIn, GitHub and WhatsApp.
* [ ] Advanced Voice Mode - Users will be able to talk to Sentient!
* [ ] Browser Use - Sentient will be able to control the browser and interact with websites to perform tasks
* [ ] Full integration at the OS level - integrate context from what you are doing right now by sharing your screen with Sentient
* [ ] Customizable Agentic Actions - users can create their own integrations using an easy-to-use interface. Users should also be able to trade custom Actions on a Marketplace.

<!-- Getting Started -->
## 	:toolbox: Getting Started

<!-- Installation -->
### :gear: Installation (Users)

If you're not interested in contributing to the project and simply want to use Sentient, download the latest release [from our website.](https://existence-sentient.vercel.app/download).
All dependencies are packaged into our installer - you do not need to download anything else. You can find installation instructions [in our docs.](https://beneficial-billboard-012.notion.site/Downloading-and-Installing-Sentient-1916d7df083c80328672c50dd0599523)

<!-- Prerequisites -->
### :bangbang: Prerequisites (Contributors)

#### The following instructions are for Linux based machines, but they remain fundamentally same for Windows & Mac

Clone the project

```bash
  git clone https://github.com/existence-master/Sentient.git
```

Go to the project directory

```bash
  cd Sentient
```

Install the following to start contributing to Sentient:
- npm: The ElectronJS frontend of the Sentient desktop app uses npm as its package manager.

  Install the latest version of NodeJS and npm from [here.](https://nodejs.org/en/download)

  After that, install all the required packages.

  ```bash
   cd ./src/interface && npm install 
  ```
  
- python: Python will be needed to run the backend. 
  Install Python [from here.](https://www.python.org/downloads/) We recommend Python 3.11.

  After that, you will need to create a virtual environment and install all required packages. This venv will need to be activated whenever you want to run any scripts on the Python backend.

  ```bash
   cd src/model && python3 -m venv venv
   cd venv/bin && source activate
   cd ../../ && pip install -r requirements.txt
  ```

- Ollama: Download and install the latest version of Ollama [from here.](https://ollama.com/)

  After that, pull the model from Ollama.
  ```bash
   ollama pull llama3.2:3b
  ```

- Neo4j Community: Download Neo4j Community Edition [from here.](https://neo4j.com/deployment-center/)

  Next, you will need to enable the APOC plugin.
  After extracting Neo4j Community Edition, navigate to the labs folder. Copy the `apoc-x.x.x-core.jar` script to the plugins folder in the Neo4j folder.
  Edit the neo4j.conf file to allow the use of APOC procedures:
  ```bash
  sudo nano /etc/neo4j/neo4j.conf
  ```
  Uncomment or add the following lines:
  ```ini
  dbms.security.procedures.unrestricted=apoc.*
  dbms.security.procedures.allowlist=apoc.*
  dbms.unmanaged_extension_classes=apoc.export=/apoc
  ```
  You can run Neo4j community using the following commands
  ```bash
    cd neo4j/bin && ./neo4j console
  ```
  While Neo4j is running, you can visit `http://localhost:7474/` to run Cypher Queries and interact with your knowledge graph.

  ⚠️ On your first run of Neo4j Community, you will need to set a username and password. **Remember this password** as you will need to add it to the `.env` file on the Python backend.

<!-- Environment Variables -->
### 🔒: Environment Variables (Contributors)

You will need the following environment variables to run the project locally. For sensitive keys like Auth0, GCP, Brave Search you can create your own accounts and populate your own keys or comment in the discussion titled ['Request Environment Variables (.env) Here'](https://github.com/existence-master/Sentient/discussions/13) if you want pre-setup keys

For the Electron Frontend, you will need to create a `.env` file in the `src/interface` folder. Populate that `.env` file with the following variables (examples given).

```.emv.template
  ELECTRON_APP_URL= "http://localhost:3000"
  APP_SERVER_URL= "http://127.0.0.1:5000"
  APP_SERVER_LOADED= "false"
  APP_SERVER_INITIATED= "false"
  NEO4J_SERVER_URL= "http://localhost:7474"
  NEO4J_SERVER_STARTED= "false"
  BASE_MODEL_REPO_ID= "llama3.2:3b"
  AUTH0_DOMAIN = "abcdxyz.us.auth0.com"
  AUTH0_CLIENT_ID = "abcd1234"
```

For the Python Backend, you will need to create a `.env` file and place it in the `src/model` folder. Populate that `.env` file with the following variables (examples given).

```.emv.template
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=abcd1234
  EMBEDDING_MODEL_REPO_ID=sentence-transformers/all-MiniLM-L6-v2
  BASE_MODEL_URL=http://localhost:11434/api/chat
  BASE_MODEL_REPO_ID=llama3.2:3b
  LINKEDIN_USERNAME=email@address.com
  LINKEDIN_PASSWORD=password123
  BRAVE_SUBSCRIPTION_TOKEN=YOUR_TOKEN_HERE
  BRAVE_BASE_URL=https://api.search.brave.com/res/v1/web/search
  GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
  GOOGLE_PROJECT_ID=YOUR_PROJECT_ID
  GOOGLE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
  GOOGLE_TOKEN_URI=https://oauth2.googleapis.com/token
  GOOGLE_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
  GOOGLE_CLIENT_SECRET=YOUR_SECRET_HERE
  GOOGLE_REDIRECT_URIS=http://localhost
  AES_SECRET_KEY=YOUR_SECRET_KEY_HERE (256 bits or 32 chars)
  AES_IV=YOUR_IV_HERE (256 bits or 32 chars)
  AUTH0_DOMAIN=abcdxyz.us.auth0.com
  AUTH0_MANAGEMENT_CLIENT_ID=YOUR_MANAGEMENT_CLIENT_ID
  AUTH0_MANAGEMENT_CLIENT_SECRET=YOUR_MANAGEMENT_CLIENT_SECRET
  APP_SERVER_PORT=5000
  AGENTS_SERVER_PORT=5001
  MEMORY_SERVER_PORT=5002
  CHAT_SERVER_PORT=5003
  SCRAPER_SERVER_PORT=5004
  UTILS_SERVER_PORT=5005
  COMMON_SERVER_PORT=5006
  AUTH_SERVER_PORT=5007
```

<!-- Run Locally -->
### :running: Run Locally (Contributors)

**Install dependencies**

Ensure that you have installed all the dependencies as outlined in the [Prerequisites Section](#bangbang-prerequisites).

**Start Neo4j**

Start Neo4j Community Edition first.

```bash
cd neo4j/bin && ./neo4j console
```

**Start the Python servers**

Example: Agents Server
```bash
  cd src/model/venv/bin/ && source activate
  cd ../../agents && sudo ./startserv.sh
```

⚠️ Be sure to modify the `startserv.sh` script with the absolute paths to your scripts and your venv.

⚠️ You will need to run all the servers manually to run the app in development mode.

Start the Electron App

```bash
  cd src/interface && npm run dev
```

`❗ You are free to package and bundle your own versions of the app that may or may not contain any modifications. However, if you do make any modifications, you must comply with the AGPL license and open-source your version as well.`

<!-- Usage -->
## :eyes: Usage

Sentient is a personal companion that remembers information about you and uses this personal context to respond to queries and perform actions for you.

### Use-Cases:

Sentient can already do a lot.

- `"Hey Sentient, help me find a restaurant in Pune based on my food preferences.`
- `What are the upcoming events in my Google Calendar?`
- `Setup a lunch meeting with tom@email.com and add it to my Calendar`
- `Create a pitch deck for my startup in Google Slides and email it to tom@email.com`
- `Help me find new hobbies in my city`

📹 [Check out our ad!](https://www.youtube.com/watch?v=Oeqmg25yqDY)

<!-- Contributing -->
## :wave: Contributing

<a href="https://github.com/existence-master/Sentient/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=existence-master/Sentient" />
</a>


Contributions are always welcome!

See the [contributing guide](https://github.com/existence-master/Sentient/blob/master/CONTRIBUTING.md) for ways to get started.


<!-- Code of Conduct -->
### :scroll: Code of Conduct

Please read the [code of conduct](https://github.com/existence-master/Sentient/blob/master/CODE_OF_CONDUCT.md)

<!-- FAQ -->
## :grey_question: FAQ

- Do you need internet to run the app?

  + No. Since the app is fully local, you do not need internet to use the core features. You only need Internet to launch the app since Auth0 needs to authenticate you with your token. Of course, you will also need internet to perform most actions like adding events to your calendar, sending emails, etc.

- Why is authentication needed?

  + We only ask for authentication so that we as developers can know who our users are. This can help us stay in touch with early users for development purposes.

- What data do you collect about me?

  + For auth, we have a standard email-password flow provided by Auth0 (also supports Google OAuth). So, the only data we collect is the email provided by users and their login history. This helps us understand how users are using the app, retention rates, daily signups and more. Read more about data collection in our [privacy policy.](https://existence-sentient.vercel.app/privacy)

- What kind of hardware do I need to run the app?

  + To run Sentient - any decent CPU (Intel Core i5 or equivalent and above), 8GB of RAM and a GPU with 4-6GB of VRAM should be enough. A GPU is required for smooth local model inference. The app only runs on Windows for now. If you run the app from source/build your own versions with a different model, these system requirements will change.

- Why open source?

  + Since the app is going to be processing a lot of your personal information, maintaining transparency of the underlying code and processes is very important. The code needs to be available for everyone to freely view how their data is being managed in the app. We also want developers to be able to contribute to Sentient - they should be able to add missing integrations or features that they feel should be a part of Sentient. They should also be able to freely make their own forks of Sentient for different use-cases, provided they abide by the GNU AGPL license and open-source their work.

- Why AGPL?

  + We intentionally decided to go with a more restrictive license, specifically AGPL, rather than a permissive license (like MIT or Apache) since we do not want any closed-source, cloud-based competitors cropping up with our code at its core. Going with AGPL is our way of staying committed to our core principles of transparency and privacy while ensuring that others who use our code also follow the same principles.

<!-- License -->
## :warning: License

Distributed under the GNU AGPL License. Check [our lisence](https://github.com/existence-master/Sentient/blob/master/LICENSE.txt) for more information.

<!-- Contact -->
## :handshake: Contact

[existence.sentient@gmail.com](existence.sentient@gmail.com)

<!-- Acknowledgments -->
## :gem: Acknowledgements

Sentient wouldn't have been possible without

 - [Ollama](https://ollama.com/)
 - [Neo4j](https://neo4j.com/)
 - [FastAPI](https://fastapi.tiangolo.com/)
 - [Meta's Llama Models](https://www.llama.com/)
 - [ElectronJS](https://www.electronjs.org/)
 - [Next.js](https://nextjs.org/)

<!-- Official Team -->
## :heavy_check_mark: Official Team

The official team behind Sentient

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section. -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tr>
     <td align="center">
       <a href="https://github.com/itsskofficial">
         <img src="https://avatars.githubusercontent.com/u/65887545?v=4?s=100" width="100px;" alt=""/>
         <br />
         <sub>
           <b>
             itsskofficial
           </b>
         </sub>
       </a>
       <br />
     </td>  
     <td align="center">
       <a href="https://github.com/kabeer2004">
         <img src="https://avatars.githubusercontent.com/u/59280736?v=4" width="100px;" alt=""/>
         <br />
         <sub>
           <b>
             kabeer2004
           </b>
         </sub>
       </a>
       <br />
     </td>  
      <td align="center">
       <a href="https://github.com/abhijeetsuryawanshi12">
         <img src="https://avatars.githubusercontent.com/u/108229267?v=4" width="100px;" alt=""/>
         <br />
         <sub>
           <b>
             abhijeetsuryawanshi12
           </b>
         </sub>
       </a>
       <br />
     </td>
  </tr>
</table>
<br />


![Powered By](./.github/assets/powered-by.png)
