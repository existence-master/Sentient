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
    <a href="https://sentient-2.gitbook.io/docs">Documentation</a>
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
  - [Philosophy](#thought_balloon-philosophy)
  - [Screenshots](#camera-screenshots)
  - [Tech Stack](#space_invader-tech-stack)
  - [Features](#dart-features)
- [Roadmap](#compass-roadmap)
- [Getting Started](#toolbox-getting-started)
  - [Prerequisites](#bangbang-prerequisites-contributors)
  - [Installation](#gear-installation-users)
  - [Environment Variables](#-environment-variables-contributors)
  - [Run Locally](#running-run-locally-contributors)
- [Usage](#eyes-usage)
- [Contributing](#wave-contributing)
  - [Code of Conduct](#scroll-code-of-conduct)
- [FAQ](#grey_question-faq)
- [License](#warning-license)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)
- [Official Team](#heavy_check_mark-official-team)

<!-- About The Project -->

## :star2: About The Project

<!-- Philosophy -->

## :thought_balloon: Philosophy

We at [Existence](https://existence.technology) believe that AI won't simply die as a fad or remain limited to an assistant. Instead, it will evolve to be a true companion of humans, and our aim with Sentient is to contribute to that future. Building a true companion requires excellent automation which in turn requires deep personalization. And if we want the AI to be completely context-aware of the user, privacy is non-negotiable. Hence, we aim to build a proactive & interactive AI companion. And we want to build it in the open, not behind closed doors, with transparency as a core pillar.

<!-- Screenshots -->

### :camera: Screenshots

<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431842199-b76c7a9a-1689-42de-93ed-5d04d6c7ad10.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDIxOTktYjc2YzdhOWEtMTY4OS00MmRlLTkzZWQtNWQwNGQ2YzdhZDEwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY2ZThhMDIyZmJkZWYxYzE5MzMyNTYzZDM5NjY0MmM3ZDc2NmJjMmYwNGU5MjUzMmJhYTE1NDU3NDhhZGIwODgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.U2Bn6mIdJF2SvXpJ9fyKe2c36-feA2wKtvQNcYjaEYY" alt="screenshot" />
  <p align="center">Context is streamed in from your apps - Sentient uses this context to 👇</p>
</div>
<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431841076-c7337318-38e2-4515-848d-df6ce9ec8685.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDEwNzYtYzczMzczMTgtMzhlMi00NTE1LTg0OGQtZGY2Y2U5ZWM4Njg1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFkYmIzYWJkMDExMmU0NzllMmZmNjU0NmUyNzIyYzJlZjUwMzM1ZDY0NjY0NjlhYTM4ODNiOGNmNDRkYzhhZTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.s87SsI2uPocqdoRQK-b_1R89ApFKnvOoVzislh77bAw" alt="screenshot" />
  <p align="center">Learn Long-Term Memories about you</p>
</div>
<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431841142-33edc431-6be9-45b3-9b9c-5262f459ede6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDExNDItMzNlZGM0MzEtNmJlOS00NWIzLTliOWMtNTI2MmY0NTllZGU2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFlOTVhMWEyODVhMWZmN2ZmMzNjMGMyZWMxZjQwYzFkNGM4OGZhZTQ4YjVkYTc5MmRhY2ZmZGQxZTBmOTY4NjUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.GUEsRDZzletVFm4uKBQhRehk4l2FhzEJuX5jFnglbZ4" alt="screenshot" />
  <p align="center">Learn Short-Term Memories about you</p>
</div>
<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431841274-ea980432-1357-451b-93d2-d952a65f4607.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDEyNzQtZWE5ODA0MzItMTM1Ny00NTFiLTkzZDItZDk1MmE2NWY0NjA3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZjMTM5OWRlMGI1ODI0Zjg4YmJiYjk2MDBmMWNjNDdhMDRjODM2YjBhNjJjY2JiMzMxMGNlM2UzYjU5OGFmYzcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.F76G4nymktipQtkQZQ_9sfmMFKiQ1AH-0hMoPWt0DQE" alt="screenshot" />
  <p align="center">Perform Actions for you, asynchronously and by combining all the different tools it needs to complete a task.</p>
</div>
<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431842176-c1ec90b6-edcc-4f9c-bc94-aa2e40b6422f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDIxNzYtYzFlYzkwYjYtZWRjYy00ZjljLWJjOTQtYWEyZTQwYjY0MjJmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRkZGMyY2Y1NTkyMDk1YjY4NWEwZjY1NDUxNWQ5NDc2NWU1OTAwZmM3ZjVjYWNmZDQzYWE1ZGNkMjJiYjQ3ZDImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.4djs4rCVqHY4L5_gshezAiMNgIcLui_eiFbZc8rsKrY" alt="screenshot" />
  <p align="center">You can also voice-call Sentient anytime for a low-latency, human-like interactive experience.</p>
</div>
<div align="center"> 
  <img src="https://private-user-images.githubusercontent.com/59280736/431842396-03af93ff-6acd-44c7-a973-dca20ac205bd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQzMTkyMTYsIm5iZiI6MTc0NDMxODkxNiwicGF0aCI6Ii81OTI4MDczNi80MzE4NDIzOTYtMDNhZjkzZmYtNmFjZC00NGM3LWE5NzMtZGNhMjBhYzIwNWJkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDEwVDIxMDE1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZmNGNkZjI5OTQ1YmFjNmYzMmIzNThiOWEyZmIyZTBiMjVlMjczNTc2NmY3MjU1NjkzOTMwNjUwYzgyZDliMzImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.XCieUKi8dB-r8H75QHWKwX7UBtC6m1NXbSFxbUV_lkI" alt="screenshot" />
  <p align="center">Your profile can also be enriched with data from other social media sites.</p>
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

- Proactive - Autonomously pulls context from your connected apps/context streams. It then uses this context to learn about you and perform actions for you.
- Self-hostable, with support for Ollama (for the text model), llamacpp for the voice models and HF-transformers for some other models.
- Full cloud-hosted version coming soon! [Join this group to be a part of the waitlist.](https://chat.whatsapp.com/IOHxuf2W8cKEuyZrMo8DOJ)
- LinkedIn, Reddit and X Integration for enriching personal profiles with even more data.
- Self-Managed Memory - Sentient can learn about the user from their interactions with it. Long-term memories are saved in a knowledge graph, while short-term memories are saved in a relational database.
- Agentic integrations for GSuite - Gmail, GCalendar, GDrive, GDocs, GSheets, GSlides support. Sentient can read context from and perform actions with all these tools. (WIP, some tools are still being expanded)
- Voice Mode - Switch to voice at any time for a smoother, low-latency, human-like interaction with Sentient.
- Web Search capabilities - Sentient can search the web to give users daily briefs, completely autonomously and also for providing additional context to answer queries (if required).
- Uni-chat: Everything happens in one chat - no need to switch between different chats for different topics. (WIP, currently uses the long and short-term memories to maintain context. We will soon introduce even more memory features.)
- Auto-updates for the Electron client ensure that you are always on the latest version.

<!-- Roadmap -->

## :compass: Roadmap

- [x] (WIP) Solving stability issues with model inference - improving structured outputs, better error handling, and more
- [x] Dual Memory - Sentient has a short-term memory and a long-term memory. The short-term memory will be stored in a relational DB and used to maintain reminders and other short-term information. Longer-term facts about the user (such as relationships, preferences, etc) will persist in the knowledge graph.
- [x] (WIP) Intent - Sentient will be able to perform actions autonomously based on triggers from the context pulled from a user's apps and triggers launched by the short-term memories. 
- [x] Memory and Task Queues - Send memory and agent operations to an async queue for being handled separate from the current conversation - improves response time.
- [x] Advanced Voice Mode - Users can talk to Sentient using only their voice.
- [x] (WIP) Better Internet Search - Internet search will include images, citations, direct links to sources and more. 
- [ ] Integrate reasoning models for planning steps when performing actions.
- [ ] Implement multi-user, cloud features with scalability in mind. Migrate to the cloud and go for a full cloud launch.
- [ ] Mobile Companion App - Mobile app will be added to allow users to access Sentient from even more devices. Even more context can be added for Sentient from the phone's push notifications, mic, etc.
- [ ] More tools! - More tools will be added in the order of user requests. Current requests - Microsoft 365, Notion, LinkedIn, GitHub and WhatsApp.
- [ ] Audio Context - adds the microphones on your devices as an additional context stream.
- [ ] Full integration at the OS level - integrate context from what you are doing right now by sharing your screen with Sentient
- [ ] Tool-specific UI - for example, a tool that retrieves stock prices will also show a graph of the past trend of that particular ticker.
- [ ] Browser/Computer Use - Sentient will be able to control a cloud VM to perform general actions for users like downloading, reading and reviewing files and more.
- [ ] Customizable Agentic Actions - users can create their own integrations using an easy-to-use interface. Users should also be able to trade custom Actions on a Marketplace.

<!-- Getting Started -->

## :toolbox: Getting Started

<!-- Installation -->

### :gear: Installation (Users)

If you're not interested in contributing to the project or self-hosting and simply want to use Sentient, join the [Early Adopters Group](https://chat.whatsapp.com/IOHxuf2W8cKEuyZrMo8DOJ). You can also join our paid waitlist for $3 - to do this, contact [@itsskofficial](https://github.com/itsskofficial). Users on the paid waitlist will be the first to get access to the full cloud version of Sentient via a closed beta.

If you are interested in contributing to the app or simply running the current latest version from source, you can proceed with the following steps 👇

<!-- Prerequisites -->

### :bangbang: Prerequisites (Contributors)

#### The following instructions are for Linux-based machines, but they remain fundamentally the same for Windows & Mac. Only things like venv configs and activations change on Windows, the rest of the process is pretty much the same.

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
   cd ./src/client && npm install
  ```

- python: Python will be needed to run the backend.
  Install Python [from here.](https://www.python.org/downloads/) We recommend Python 3.11.

  After that, you will need to create a virtual environment and install all required packages. This venv will need to be activated whenever you want to run the Python server (backend).

  ```bash
   cd src/server && python3 -m venv venv
   cd venv/bin && source activate
   cd ../../ && pip install -r requirements.txt
  ```

  `⚠️ If you get a numpy dependency error while installing the requirements, first install the requirements with the latest numpy version (2.x). After the installation of requirements completes, install a numpy 1.x version (backend has been tested and works successfully on numpy 1.26.4) and you will be ready to go. This is probably not the best practise, but this works for now.`

  `⚠️ If you intend to use Advanced Voice Mode, you MUST download and install llama-cpp-python with CUDA support (if you have an NVIDIA GPU) using the commented out pip command in the requirements.txt file. Otherwise, simply download and install the llama-cpp-python package with pip for simple CPU-only support. This line is commented out in the requirements file to allow users to download and install the appropriate version based on their preference (CPU only/GPU accelerated).`

- Ollama: Download and install the latest version of Ollama [from here.](https://ollama.com/)

  After that, pull the model you wish to use from Ollama. For example,

  ```bash
   ollama pull llama3.2:3b
  ```

  `⚠️ By default, the backend is configured with Llama 3.2 3B. We found this SLM to be really versatile and works really well for our usage, as compared to other SLMs. However a lot of new SLMs like Cogito are being dropped everyday so we will probably be changing the model soon. If you wish to use a different model, simply find all the places where llama3.2:3b has been set in the Python backend scripts and change it to the tag of the model you have pulled from Ollama.`

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

  `⚠️ On your first run of Neo4j Community, you will need to set a username and password. **Remember this password** as you will need to add it to the .env file on the Python backend.`

- Download the Voice Model (Orpheus TTS 3B)

  For using Advanced Voice Mode, you need to manually download [this model](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) from Huggingface. Whisper is automatically downloaded by Sentient via fasterwhisper.

  The model linked above is a Q4 quantization of the Orpheus 3B model. If you have even more VRAM at your disposal, you can go for the [Q8 quant](https://huggingface.co/Mungert/orpheus-3b-0.1-ft-GGUF).

  Download the GGUF files - these models are run using llama-cpp-python.

  Place the model files here:
  `src/server/voice/models`

  and ensure that the correct model name is set in the Python scripts on the backend. By default, the app is configured to use the 8-bit quant using the same name that it has when you download it from HuggingFace.

  `⚠️ If you do not have enough VRAM and voice mode is not that important to you, you can comment out/remove the voice mode loading functionality in the main app.py located at src/server/app/app.py`

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
```

`⚠️ If you face some issues with Auth0 setup, please contact us via our Whatsapp Group or reach out to one of the lead contributors [@Kabeer2004](https://github.com/Kabeer2004), [@itsskofficial](https://github.com/itsskofficial) or [@abhijeetsuryawanshi12](https://github.com/abhijeetsuryawanshi12)`

<!-- Run Locally -->

### :running: Run Locally (Contributors)

**Install dependencies**

Ensure that you have installed all the dependencies as outlined in the [Prerequisites Section](#bangbang-prerequisites).

**Start Neo4j**

Start Neo4j Community Edition first.

```bash
cd neo4j/bin && ./neo4j console
```

**Start the Python backend server.**

```bash
  cd src/server/venv/bin/ && source activate
  cd ../../ && python -m server.app.app
```

Once the Python server has fully started up, start the Electron client.

```bash
  cd src/interface && npm run dev
```

`❗ You are free to package and bundle your own versions of the app that may or may not contain any modifications. However, if you do make any modifications, you must comply with the AGPL license and open-source your version as well.`

<!-- Usage -->

## :eyes: Usage

Sentient is a proactive companion that pulls context from the different apps you use and uses that context to remember new information about you, perform tasks for you like sending emails, creating docs and more. The goal of this personal companion is to take over the mundane tasks in your life to let you focus on what matters.

### Use-Cases:

Sentient can also do a lot based on simple user commands.

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

- When will the cloud version launch?

  - We are working as fast as we can to bring it to life! Join our [WhatsApp Community](https://chat.whatsapp.com/IOHxuf2W8cKEuyZrMo8DOJ) to get daily updates and more!

- What data do you collect about me?

  - For auth, we have a standard email-password flow provided by Auth0 (also supports Google OAuth). So, the only data we collect is the email provided by users and their login history. This helps us understand how users are using the app, retention rates, daily signups and more. Read more about data collection in our [privacy policy.](https://existence-sentient.vercel.app/privacy).

- What kind of hardware do I need to run the app locally/self-host it?

  - To run Sentient - any decent CPU (Intel Core i5 or equivalent and above), 8GB of RAM and a GPU with 4-6GB of VRAM should be enough for text only. For voice, additional VRAM will be required based on the quant you choose for Orpheus 3B. A GPU is necessary for fast local model inference. You can self-host/run locally on Windows, Linux or Mac.

- Why open source?

  - Since the app is going to be processing a lot of your personal information, maintaining transparency of the underlying code and processes is very important. The code needs to be available for everyone to freely view how their data is being managed in the app. We also want developers to be able to contribute to Sentient - they should be able to add missing integrations or features that they feel should be a part of Sentient. They should also be able to freely make their own forks of Sentient for different use-cases, provided they abide by the GNU AGPL license and open-source their work. 

- Why AGPL?

  - We intentionally decided to go with a more restrictive license, specifically AGPL, rather than a permissive license (like MIT or Apache) since we do not want any other closed-source, cloud-based competitors cropping up with our code at its core. Going with AGPL is our way of staying committed to our core principles of transparency and privacy while ensuring that others who use our code also follow the same principles.

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
