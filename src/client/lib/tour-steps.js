// src/client/lib/tour-steps.js

export const tourSteps = [
	// Step 0: Welcome Mat (Modal)
	{
		type: "modal",
		title: "Welcome to BUYASOUL! Let's see your AI in action.",
		body: "This quick, interactive tour will show you how I handle everything from simple commands to complex projects. You'll get to see the full lifecycle of an automated task.",
		buttons: [
			{
				label: "Start Tour",
				primary: true
			},
			{ label: "Skip for now" }
		]
	},
	// Step 1: Send Chat Message
	{
		type: "tooltip",
		path: "/chat",
		selector: "[data-tour-id='chat-input-area']",
		title: "Step 1/7: Give a Command",
		body: "For quick actions, you can tell me what to do. I'll handle it and reply here. After you send the message, I'll prepare the next one.",
		instruction:
			"Let's start with a simple greeting. Click the send button.",
		isWaitingForAction: true
	},
	// Step 2: Go to Tasks Page
	{
		type: "tooltip",
		path: "/chat",
		selector: "[data-tour-id='sidebar-tasks-icon']",
		title: "Step 2/7: Delegating Complex Work",
		body: "That was a simple task. For bigger goals with multiple steps, I create a project on the Tasks page that you can track. Let's see a simulation of how that works.",
		instruction: "Click the Tasks icon to continue.",
		action: { type: "navigate", targetPath: "/tasks" }
	},
	// Step 3: Create Task Button
	{
		type: "tooltip",
		path: "/tasks",
		selector: "[data-tour-id='create-task-button']",
		title: "Step 3/7: Creating a Complex Task",
		body: "Let's create a more complex, multi-step task. You can start by describing your goal in plain English.",
		instruction: "Click the 'Create Task' button to open the composer.",
		isWaitingForAction: true,
		initialDelay: 500
	},
	// Step 4: Composer
	{
		type: "tooltip",
		path: "/tasks",
		selector: "[data-tour-id='task-composer']",
		title: "Step 4/7: Describe Your Goal",
		body: "I've pre-filled the composer with a goal. I will analyze this and create a plan to achieve it.",
		instruction: "Click 'Create Task' to see me get to work.",
		isWaitingForAction: true,
		initialDelay: 500
	},
	// Step 5: Task Lifecycle Simulation
	{
		type: "tooltip",
		path: "/tasks",
		selector: "[data-tour-id='demo-task-card']",
		title: "Step 5/7: Task Lifecycle",
		body: "This is a simulation of a long-form task. Follow the steps to see how I handle complex goals.",
		instruction: "",
		initialDelay: 500
	},
	// Step 6: Go to Integrations Page
	{
		type: "tooltip",
		path: "/tasks",
		selector: "[data-tour-id='sidebar-integrations-icon']",
		title: "Step 6/7: Connect Your Apps",
		body: "None of this works without connecting your apps. This is where you can manage connections to services like Gmail, Calendar, and more.",
		instruction: "Click the Integrations icon to see.",
		action: { type: "navigate", targetPath: "/integrations" }
	},
	// Step 7: Tour Complete
	{
		type: "modal",
		title: "You're Ready to Go!",
		body: "You've now seen how Sentient can handle immediate commands, orchestrate complex projects, and automate your work with workflows. You can replay the task simulation anytime from the Help menu.",
		buttons: [{ label: "Finish Tour", primary: true }]
	}
]

export const chatSubSteps = [
	{
		prefill: "Hi Sentient!",
		instruction: "Let's start with a simple greeting. Click the send button."
	},
	{
		prefill: "Send an email to existence.sentient@gmail.com",
		instruction:
			"Great! Now, let's ask it to perform an action. Click send."
	},
	{
		prefill:
			"Send me a daily brief of my unread emails on whatsapp every morning at 8"
	}
]

export const taskSubSteps = [
	{
		title: "Step 5/7: Planning",
		body_list:
			"The task has been created and is now in the 'Planning' stage. I'm breaking down your goal into a series of steps.",
		body_panel:
			"In the details panel, you can see the execution log. It shows that I'm currently in the planning phase.",
		button: "Simulate Next Step"
	},
	{
		title: "Step 5/7: Taking Action",
		body_list:
			"I've created the first sub-task: to email Kabeer. The main task status is now 'Processing'.",
		body_panel:
			"I'll search for Kabeer's contact details and send the email automatically. The new sub-task appears here.",
		button: "Simulate Next Step"
	},
	{
		title: "Step 5/7: Waiting Intelligently",
		body_list:
			"Now, I'll wait for Kabeer to reply. The main task status is now 'Waiting'. I won't waste resources; I'll pause and check back later.",
		body_panel:
			"The log shows I'm waiting. This could be for a few hours in a real task. Let's fast-forward time.",
		button: "Simulate Next Step"
	},
	{
		title: "Step 5/7: Following Up",
		body_list:
			"The waiting period is over. I've created a new sub-task to check the email thread for a response. Let's assume Kabeer replied.",
		body_panel:
			"The new sub-task to check the email thread is now visible. Once this is complete, I'll know what to do next.",
		button: "Simulate Next Step"
	},
	{
		title: "Step 5/7: Finalizing the Goal",
		body_list:
			"Kabeer suggested a time. I'm now creating the final sub-task to schedule the event in your calendar.",
		body_panel:
			"The final sub-task to create the calendar event has been added. The goal is almost complete.",
		button: "Simulate Next Step"
	},
	{
		title: "Step 5/7: Task Completed!",
		body_list:
			"Success! All sub-tasks are done, and the main goal is achieved. The task is now marked as 'Completed'.",
		body_panel:
			"The task is complete. You can review the full history of sub-tasks and logs at any time.",
		button: "Next"
	}
]
