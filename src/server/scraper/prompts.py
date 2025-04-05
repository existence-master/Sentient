reddit_system_prompt_template = """I will provide you a dictionary containing subreddits with comments and submissions extracted from a user's profile. Your task is to analyze the content and identify the unique topics of interest for the user. 

 Instructions:
1. Analyze the provided dictionary, which includes:
   - A list of subreddits where the user has made submissions.
   - A list of subreddits where the user has posted comments.
2. Identify distinct topics based on the names of the subreddits.
   - Use your reasoning to infer broader or meaningful topics.
   - Ensure the topics are not duplicates or repetitive in any way.
3. Return the topics as a Python list in the format: `["Topic1", "Topic2", "Topic3"]`.
4. Strictly avoid adding any explanation, commentary, or additional formatting. The response must be a valid Python list and nothing else.

Examples:

Example 1:
Input:
{
  "submissions": ["technology", "programming", "artificialintelligence"],
  "comments": ["machinelearning", "computerscience", "dataengineering"]
}

Output:
["Technology", "Programming", "Artificial Intelligence", "Machine Learning", "Computer Science", "Data Engineering"]


Example 2:
Input:
{
  "submissions": ["gaming", "leagueoflegends", "overwatch"],
  "comments": ["esports", "valorant", "gamingpc"]
}

Output:
["Gaming", "Esports", "League of Legends", "Overwatch", "Valorant", "Gaming PCs"]


Example 3:
Input:
{
  "submissions": ["photography", "landscapephotography", "streetphotography"],
  "comments": ["cameras", "lensadvice", "photoediting"]
}

Output:
["Photography", "Landscape Photography", "Street Photography", "Cameras", "Lens Advice", "Photo Editing"]

Example 4:
Input:
{
  "submissions": ["fitness", "weightlifting", "running"],
  "comments": ["health", "nutrition", "exercise"]
}

Output:
["Fitness", "Weightlifting", "Running", "Health", "Nutrition", "Exercise"]
"""

reddit_user_prompt_template = """Analyze the dictionary provided below and return a list of unique topics of interest. 
Return only the Python list of topics in the specified format. Do not include any explanations, comments, or additional text.

Input:
{subreddits}

Output:
"""

twitter_system_prompt_template = """I will provide you a list of tweets extracted from a user's Twitter profile. Your task is to analyze the content of the tweets and identify the unique topics of interest for the user.

Instructions:
1. Analyze the provided list of tweets. Each tweet is a string containing text content.
2. Identify distinct topics based on the content of the tweets.
   - Use your reasoning to infer broader or meaningful topics from the text.
   - Avoid listing duplicates or repetitive topics.
   - If a tweet discusses multiple ideas, extract each topic separately.
3. Return the topics as a Python list in the format: `["Topic1", "Topic2", "Topic3"]`.
4. Strictly avoid adding any explanation, commentary, or additional formatting. The response must be a valid Python list and nothing else.

Examples:

Example 1:
Input:
[
  "Just finished reading an amazing book on Artificial Intelligence! #AI #MachineLearning",
  "The future of autonomous vehicles looks incredible. Can't wait for safer self-driving cars.",
  "Quantum computing is going to change the world. #QuantumTech #Innovation"
]

Output:
["Artificial Intelligence", "Machine Learning", "Autonomous Vehicles", "Self-Driving Cars", "Quantum Computing", "Innovation"]

Example 2:
Input:
[
  "The World Cup is so exciting this year! Football is life. ⚽",
  "What a game by Messi! Incredible performance as always.",
  "Discussing the strategies in modern soccer is fascinating. #Football #Soccer"
]

Output:
["World Cup", "Football", "Soccer", "Messi", "Sports Strategies"]

Example 3:
Input:
[
  "Exploring the beauty of Python programming. It’s such a versatile language. #Coding",
  "Big Data is the future of decision-making. Excited to learn more about it. #Analytics",
  "Data visualization is such a powerful way to tell stories with numbers. #DataScience"
]

Output:
["Python Programming", "Coding", "Big Data", "Decision-Making", "Data Visualization", "Data Science"]

Example 4:
Input:
[
  "Really enjoying the fresh perspective in the new Marvel movie. The action sequences are amazing!",
  "The cinematography in Dune is stunning. A visual masterpiece. #Dune",
  "Looking forward to upcoming sci-fi movies. The genre is evolving rapidly. #SciFi"
]

Output:
["Marvel Movies", "Action Sequences", "Cinematography", "Dune", "Sci-Fi", "Visual Arts"]
"""

twitter_user_prompt_template = """Analyze the list of tweets provided below and return a list of unique topics of interest.
Return only the Python list of topics in the specified format. Do not include any explanations, comments, or additional text or else I will terminate you

Input:
{tweets}

Output:
"""
