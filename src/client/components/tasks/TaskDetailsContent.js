"use client"

import React, { useState, useEffect, useRef } from "react"
import toast from "react-hot-toast"
import { cn } from "@utils/cn"
import { taskStatusColors, priorityMap } from "./constants"
import {
	IconGripVertical,
	IconPlus,
	IconSparkles,
	IconUser,
	IconX,
	IconLoader,
	IconSend,
	IconInfoCircle,
	IconLink,
	IconWorldSearch,
	IconChevronRight
} from "@tabler/icons-react"
import ScheduleEditor from "@components/tasks/ScheduleEditor"
import ExecutionUpdate from "./ExecutionUpdate"
import ChatBubble from "@components/ChatBubble"
import { TextShimmer } from "@components/ui/text-shimmer"
import CollapsibleSection from "./CollapsibleSection"
import FileCard from "@components/FileCard"
import ReactMarkdown from "react-markdown"

// New component for handling clarification questions
const QnaSection = ({ questions, task, onAnswerClarifications }) => {
	const [answers, setAnswers] = useState({})
	const [isSubmitting, setIsSubmitting] = useState(false)
	const isInputMode = task.status === "clarification_pending"

	const handleAnswerChange = (questionId, text) => {
		setAnswers((prev) => ({ ...prev, [questionId]: text }))
	}

	const handleSubmit = async () => {
		const unansweredQuestions = questions.filter(
			(q) => !answers[q.question_id]?.trim()
		)
		if (unansweredQuestions.length > 0) {
			toast.error("Please answer all questions before submitting.")
			return
		}

		setIsSubmitting(true)
		const answersPayload = Object.entries(answers).map(
			([question_id, answer_text]) => ({
				question_id,
				answer_text
			})
		)
		// This function is passed down from the main page and includes closing the panel
		await onAnswerClarifications(task.task_id, answersPayload)
		setIsSubmitting(false) // This might not be reached if the component unmounts
	}

	return (
		<div>
			<h4 className="font-semibold text-neutral-300 mb-2">
				Clarifying Questions
			</h4>
			<div
				className={cn(
					"space-y-4 p-4 rounded-lg border",
					isInputMode
						? "bg-yellow-500/10 border-yellow-500/20"
						: "bg-neutral-800/20 border-neutral-700/50"
				)}
			>
				{questions.map((q, index) => (
					<div key={q.question_id || index}>
						<label className="block text-sm font-medium text-neutral-300 mb-2">
							{q.text}
						</label>
						{isInputMode ? (
							<textarea
								value={answers[q.question_id] || ""}
								onChange={(e) =>
									handleAnswerChange(
										q.question_id,
										e.target.value
									)
								}
								rows={2}
								className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm text-white transition-colors focus:border-yellow-400 focus:ring-0"
								placeholder="Your answer..."
							/>
						) : (
							<p className="text-sm text-neutral-100 p-2 bg-neutral-900/50 rounded-md whitespace-pre-wrap">
								{q.answer || (
									<span className="italic text-neutral-500">
										No answer provided.
									</span>
								)}
							</p>
						)}
					</div>
				))}
				{isInputMode && (
					<div className="flex justify-end">
						<button
							onClick={handleSubmit}
							disabled={isSubmitting}
							className="px-4 py-2 text-sm font-semibold bg-yellow-400 text-black rounded-md hover:bg-yellow-300 disabled:opacity-50 flex items-center gap-2"
						>
							{isSubmitting && (
								<IconLoader
									size={16}
									className="animate-spin"
								/>
							)}
							{isSubmitting ? "Submitting..." : "Submit Answers"}
						</button>
					</div>
				)}
			</div>
		</div>
	)
}

const LongFormPlanSection = ({ plan, onSelectTask }) => {
	if (!plan || plan.length === 0) {
		return (
			<div>
				<h4 className="font-semibold text-neutral-300 mb-2">Plan</h4>
				<p className="text-sm text-neutral-500">
					The orchestrator is currently generating the initial plan.
				</p>
			</div>
		)
	}

	return (
		<div>
			<h4 className="font-semibold text-neutral-300 mb-2">
				Dynamic Plan
			</h4>
			<div className="space-y-3">
				{plan.map((step, index) => (
					<div
						key={step.step_id || index}
						className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50"
					>
						<p className="text-sm font-medium text-neutral-200">
							{step.description}
						</p>
						<div className="text-xs text-neutral-400 mt-2 flex items-center justify-between">
							<span>
								Status:{" "}
								<span className="font-semibold capitalize">
									{step.status}
								</span>
							</span>
							{step.sub_task_id && (
								<button
									onClick={() =>
										onSelectTask({
											task_id: step.sub_task_id
										})
									}
									className="text-blue-400 hover:underline flex items-center gap-1"
								>
									View Sub-task <IconChevronRight size={14} />
								</button>
							)}
						</div>
						{step.result && (
							<div className="mt-2 pt-2 border-t border-neutral-700">
								<p className="text-xs font-semibold text-neutral-300 mb-1">
									Result:
								</p>
								<pre className="text-xs bg-neutral-900 p-2 rounded-md whitespace-pre-wrap max-h-40 overflow-auto custom-scrollbar">
									{JSON.stringify(step.result, null, 2)}
								</pre>
							</div>
						)}
					</div>
				))}
			</div>
		</div>
	)
}

const LongFormQnaSection = ({ requests, task, onAnswer }) => {
	const [answers, setAnswers] = useState({})
	const [isSubmitting, setIsSubmitting] = useState(null) // store request_id being submitted

	const pendingRequests = requests.filter((r) => r.status === "pending")
	if (pendingRequests.length === 0) return null

	const handleAnswerChange = (requestId, text) => {
		setAnswers((prev) => ({ ...prev, [requestId]: text }))
	}

	const handleSubmit = async (requestId) => {
		const answerText = answers[requestId]
		if (!answerText || !answerText.trim()) {
			toast.error("Please provide an answer.")
			return
		}
		setIsSubmitting(requestId)
		await onAnswer(task.task_id, requestId, answerText)
		setIsSubmitting(null)
	}

	return (
		<div>
			<h4 className="font-semibold text-neutral-300 mb-2">
				Action Required
			</h4>
			<div className="space-y-4 p-4 rounded-lg border bg-yellow-500/10 border-yellow-500/20">
				{pendingRequests.map((req) => (
					<div key={req.request_id}>
						<label className="block text-sm font-medium text-neutral-300 mb-2 whitespace-pre-wrap">
							{req.question}
						</label>
						<textarea
							value={answers[req.request_id] || ""}
							onChange={(e) =>
								handleAnswerChange(
									req.request_id,
									e.target.value
								)
							}
							rows={3}
							className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm text-white transition-colors focus:border-yellow-400 focus:ring-0"
							placeholder="Your answer..."
						/>
						<div className="flex justify-end mt-2">
							<button
								onClick={() => handleSubmit(req.request_id)}
								disabled={isSubmitting === req.request_id}
								className="px-4 py-2 text-sm font-semibold bg-yellow-400 text-black rounded-md hover:bg-yellow-300 disabled:opacity-50 flex items-center gap-2"
							>
								{isSubmitting === req.request_id && (
									<IconLoader
										size={16}
										className="animate-spin"
									/>
								)}
								{isSubmitting === req.request_id
									? "Submitting..."
									: "Submit Answer"}
							</button>
						</div>
					</div>
				))}
			</div>
		</div>
	)
}

const TaskChatSection = ({ task, onSendChatMessage }) => {
	const [message, setMessage] = useState("")
	const chatEndRef = React.useRef(null)

	useEffect(() => {
		chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
	}, [task.chat_history])

	const handleSend = () => {
		if (message.trim()) {
			onSendChatMessage(task.task_id, message)
			setMessage("")
		}
	}

	return (
		<div className="mt-6 pt-6 border-t border-neutral-800">
			<h4 className="font-semibold text-neutral-300 mb-4">
				Task Conversation
			</h4>
			<div className="space-y-4 max-h-64 overflow-y-auto custom-scrollbar pr-2">
				{(task.chat_history || []).map((msg, index) => (
					<ChatBubble
						key={index}
						role={msg.role}
						content={msg.content}
						message={msg}
					/>
				))}
				<div ref={chatEndRef} />
			</div>
			<div className="mt-4 flex items-center gap-2">
				<input
					type="text"
					value={message}
					onChange={(e) => setMessage(e.target.value)}
					onKeyDown={(e) => e.key === "Enter" && handleSend()}
					placeholder="Ask for changes or follow-ups..."
					className="flex-grow p-2 bg-neutral-800 border border-neutral-700 rounded-lg text-sm"
				/>
				<button
					onClick={handleSend}
					className="p-2 bg-blue-600 rounded-lg text-white hover:bg-blue-500 disabled:opacity-50"
					disabled={!message.trim()}
				>
					<IconSend size={16} />
				</button>
			</div>
		</div>
	)
}

const TaskResultDisplay = ({ result }) => {
	if (!result) return null

	// Handle simple string results for backward compatibility
	if (typeof result === "string") {
		return (
			<div>
				<h4 className="font-semibold text-neutral-300 mb-2">Result</h4>
				<div className="text-sm bg-neutral-800/50 p-3 rounded-lg whitespace-pre-wrap border border-neutral-700/50">
					<ReactMarkdown>{result}</ReactMarkdown>
				</div>
			</div>
		)
	}

	const { summary, links_created, links_found, files_created, tools_used } =
		result

	return (
		<div className="space-y-4">
			<h4 className="font-semibold text-neutral-300">Final Report</h4>

			{summary && (
				<div className="text-sm bg-neutral-800/50 p-4 rounded-lg border border-neutral-700/50">
					<ReactMarkdown className="prose prose-sm prose-invert">
						{summary}
					</ReactMarkdown>
				</div>
			)}

			{(links_created?.length > 0 || links_found?.length > 0) && (
				<CollapsibleSection title="Relevant Links">
					<div className="space-y-2 pt-2">
						{links_created.map((link, i) => (
							<div
								key={`created-${i}`}
								className="flex items-center gap-2 text-sm p-2 bg-neutral-800/30 rounded-md"
							>
								<IconLink size={16} className="text-blue-400" />
								<a
									href={link.url}
									target="_blank"
									rel="noopener noreferrer"
									className="text-blue-400 hover:underline truncate"
								>
									{link.description || link.url}
								</a>
							</div>
						))}
						{links_found.map((link, i) => (
							<div
								key={`found-${i}`}
								className="flex items-center gap-2 text-sm p-2 bg-neutral-800/30 rounded-md"
							>
								<IconWorldSearch
									size={16}
									className="text-green-400"
								/>
								<a
									href={link.url}
									target="_blank"
									rel="noopener noreferrer"
									className="text-green-400 hover:underline truncate"
								>
									{link.description || link.url}
								</a>
							</div>
						))}
					</div>
				</CollapsibleSection>
			)}

			{files_created?.length > 0 && (
				<CollapsibleSection title="Files Created">
					<div className="space-y-2 pt-2">
						{files_created.map((file, i) => (
							<FileCard
								key={`file-${i}`}
								filename={file.filename}
							/>
						))}
					</div>
				</CollapsibleSection>
			)}

			{tools_used?.length > 0 && (
				<CollapsibleSection title="Tools Used">
					<div className="flex flex-wrap gap-2 pt-2">
						{tools_used.map((tool) => (
							<span
								key={tool}
								className="bg-neutral-700 text-xs font-medium px-2 py-1 rounded-full"
							>
								{tool}
							</span>
						))}
					</div>
				</CollapsibleSection>
			)}
		</div>
	)
}

const SwarmDetailsSection = ({ swarmDetails }) => {
	if (!swarmDetails) return null

	const {
		goal,
		total_agents = 0,
		completed_agents = 0,
		progress_updates = [],
		aggregated_results = []
	} = swarmDetails

	const progress =
		total_agents > 0 ? (completed_agents / total_agents) * 100 : 0

	return (
		<div className="space-y-4">
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Swarm Goal
				</label>
				<div className="bg-neutral-800/50 p-3 rounded-lg text-sm text-neutral-300">
					{goal}
				</div>
			</div>
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Swarm Progress
				</label>
				<div className="bg-neutral-800/50 p-3 rounded-lg space-y-2">
					<div className="flex justify-between items-center text-xs font-mono text-neutral-300">
						<span>
							{completed_agents} / {total_agents} Agents Complete
						</span>
						<span>{Math.round(progress)}%</span>
					</div>
					<div className="w-full bg-neutral-700 rounded-full h-2.5">
						<div
							className="bg-blue-500 h-2.5 rounded-full"
							style={{ width: `${progress}%` }}
						></div>
					</div>
				</div>
			</div>
			<CollapsibleSection title="Live Log">
				<div className="space-y-2 bg-neutral-800/50 p-3 rounded-lg max-h-60 overflow-y-auto custom-scrollbar">
					{progress_updates.map((update, index) => (
						<div
							key={index}
							className="text-xs font-mono text-neutral-400"
						>
							<span className="text-neutral-500">
								[
								{new Date(
									update.timestamp
								).toLocaleTimeString()}
								]
							</span>{" "}
							[{update.status}] {update.message}
						</div>
					))}
				</div>
			</CollapsibleSection>
			{aggregated_results.length > 0 && (
				<CollapsibleSection title="Aggregated Results">
					<pre className="text-xs bg-neutral-800/50 p-3 rounded-lg whitespace-pre-wrap font-mono border border-neutral-700/50 max-h-96 overflow-auto custom-scrollbar">
						{JSON.stringify(aggregated_results, null, 2)}
					</pre>
				</CollapsibleSection>
			)}
		</div>
	)
}

const TaskDetailsContent = ({
	task,
	isEditing,
	editableTask,
	handleFieldChange,
	handleScheduleChange,
	handleAddStep,
	handleRemoveStep,
	handleStepChange,
	allTools,
	integrations,
	onSendChatMessage,
	onAnswerClarifications,
	onAnswerLongFormClarification,
	onSelectTask
}) => {
	if (!task) {
		return null
	}

	const displayTask = isEditing ? editableTask : task
	const statusInfo =
		taskStatusColors[displayTask.status] || taskStatusColors.default
    const orchestratorStatus =
        displayTask.task_type === "long_form"
            ? displayTask.orchestrator_state?.current_state
            : null

	const priorityInfo =
		priorityMap[displayTask.priority] || priorityMap.default
	let runs = displayTask.runs || []

	if (
		runs.length === 0 &&
		(displayTask.result ||
			displayTask.error ||
			displayTask.plan?.length > 0 ||
			displayTask.progress_updates?.length > 0)
	) {
		runs.push({
			run_id: "legacy",
			status: displayTask.status,
			plan: displayTask.plan,
			clarifying_questions: displayTask.clarifying_questions,
			progress_updates: displayTask.progress_updates,
			result: displayTask.result,
			error: displayTask.error
		})
	}

	return (
		<div className="space-y-6">
			{displayTask.task_type === "long_form" && (
				<LongFormPlanSection
					plan={displayTask.dynamic_plan}
					onSelectTask={onSelectTask}
				/>
			)}
			{displayTask.task_type === "long_form" &&
				displayTask.orchestrator_state?.current_state === "SUSPENDED" &&
				displayTask.clarification_requests?.some(
					(r) => r.status === "pending"
				) && (
					<LongFormQnaSection
						requests={displayTask.clarification_requests}
						task={displayTask}
						onAnswer={onAnswerLongFormClarification} // new prop
					/>
				)}

			{displayTask.clarifying_questions &&
				displayTask.clarifying_questions.length > 0 && (
					<QnaSection
						questions={displayTask.clarifying_questions}
						task={displayTask}
						onAnswerClarifications={onAnswerClarifications}
					/>
				)}
			{/* --- SWARM DETAILS (if applicable) --- */}
			{displayTask.task_type === "swarm" && (
				<div>
					<label className="text-sm font-medium text-neutral-400 block mb-2">
						Researched Context
					</label>
					<div className="bg-neutral-800/50 p-3 rounded-lg text-sm text-neutral-300 whitespace-pre-wrap border border-neutral-700/50">
						<ReactMarkdown className="prose prose-sm prose-invert">
							{displayTask.found_context}
						</ReactMarkdown>
					</div>
				</div>
			)}

			{/* --- META INFO & ASSIGNEE --- */}
			<div className="w-full">
				<div>
					<label className="text-sm font-medium text-neutral-400 block mb-2">
						Meta
					</label>
					<div className="flex w-full items-center gap-4 text-sm bg-neutral-800/50 p-3 rounded-lg">
						<span className="text-sm text-neutral-400">
							Status:
						</span>
						<span
							className={cn(
								"font-semibold w-full py-0.5 px-2 rounded-full text-xs flex items-center gap-1",
								statusInfo.color,
								statusInfo.border.replace("border-", "bg-") +
									"/20"
							)}
						>
							<statusInfo.icon size={12} />
							{statusInfo.label}
                            {orchestratorStatus && (
                                <span className="text-neutral-500 font-normal italic">
                                    ({orchestratorStatus})
                                </span>
                            )}
						</span>
						<div className="w-px h-4 bg-neutral-700"></div>
						<span className="text-sm text-neutral-400">
							Priority:
						</span>
						{isEditing ? (
							<select
								value={editableTask.priority}
								onChange={(e) =>
									handleFieldChange(
										"priority",
										Number(e.target.value)
									)
								}
								className="bg-neutral-700/50 border border-neutral-600 rounded-md px-2 py-1 text-xs appearance-none"
							>
								<option value={0}>High</option>
								<option value={1}>Medium</option>
								<option value={2}>Low</option>
							</select>
						) : (
							<span
								className={cn(
									"font-semibold w-full",
									priorityInfo.color
								)}
							>
								{priorityInfo.label}
							</span>
						)}
					</div>
				</div>
			</div>

			{/* --- DESCRIPTION --- */}
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Description
				</label>
				{isEditing ? (
					<textarea
						value={editableTask.description}
						onChange={(e) =>
							handleFieldChange("description", e.target.value)
						}
						className="w-full p-3 bg-neutral-800/50 border border-neutral-700 rounded-lg transition-colors focus:border-[var(--color-accent-blue)]"
						rows={4}
						placeholder="Detailed task description..."
					/>
				) : (
					<div className="bg-neutral-800/50 p-3 rounded-lg text-sm text-neutral-300 whitespace-pre-wrap">
						{displayTask.description || "No description provided."}
					</div>
				)}
			</div>

			{/* --- SCHEDULE --- */}
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Schedule
				</label>
				{isEditing ? (
					<ScheduleEditor
						schedule={
							editableTask.schedule || {
								type: "once",
								run_at: null
							}
						}
						setSchedule={handleScheduleChange}
					/>
				) : displayTask.schedule ? (
					<div className="bg-neutral-800/50 p-3 rounded-lg text-sm">
						{displayTask.schedule.type === "recurring"
							? `Recurring: ${displayTask.schedule.frequency} on ${displayTask.schedule.days?.join(", ")} at ${displayTask.schedule.time}`
							: `Once: ${displayTask.schedule.run_at ? new Date(displayTask.schedule.run_at).toLocaleString() : "ASAP"}`}
					</div>
				) : (
					<p className="text-sm text-neutral-500">Not scheduled.</p>
				)}
			</div>

			{/* --- PLAN & OUTCOME --- */}
			{isEditing ? (
				<div className="space-y-3">
					<label className="text-sm font-medium text-neutral-300">
						Plan Steps
					</label>
					{(editableTask.plan || []).map((step, index) => (
						<div
							key={index}
							className="flex items-center gap-2 p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50"
						>
							<IconGripVertical className="h-5 w-5 text-neutral-500 cursor-grab flex-shrink-0" />
							<select
								value={step.tool}
								onChange={(e) =>
									handleStepChange(
										index,
										"tool",
										e.target.value
									)
								}
								className="w-1/3 p-2 bg-neutral-700 border border-neutral-600 rounded-md text-sm appearance-none"
							>
								<option value="">Select tool...</option>
								{allTools.map((tool) => (
									<option key={tool.name} value={tool.name}>
										{tool.display_name}
									</option>
								))}
							</select>
							<input
								type="text"
								value={step.description}
								onChange={(e) =>
									handleStepChange(
										index,
										"description",
										e.target.value
									)
								}
								className="flex-grow p-2 bg-neutral-700 border border-neutral-600 rounded-md text-sm"
								placeholder="Step description..."
							/>
							<button
								onClick={() => handleRemoveStep(index)}
								className="p-2 text-red-400 hover:bg-red-500/10 rounded-full flex-shrink-0"
							>
								<IconX size={16} />
							</button>
						</div>
					))}
					<button
						onClick={handleAddStep}
						className="flex items-center gap-1.5 text-xs py-1.5 px-3 rounded-full bg-neutral-700 hover:bg-neutral-600 font-medium"
					>
						<IconPlus size={14} /> Add Step
					</button>
				</div>
			) : (
				runs.map((run) => (
					<div
						key={run.run_id || "default-run"}
						className="space-y-6 border-t border-neutral-800 pt-6 first:border-t-0 first:pt-0"
					>
						{run.plan && run.plan.length > 0 && (
							<div>
								<h4 className="font-semibold text-neutral-300 mb-2">
									Plan
								</h4>
								<div className="space-y-2">
									{run.plan.map((step, index) => (
										<div
											key={index}
											className="flex items-start gap-3 p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50"
										>
											<div className="flex-shrink-0 w-5 h-5 bg-neutral-700 rounded-full flex items-center justify-center text-xs font-bold">
												{index + 1}
											</div>
											<div>
												<p className="text-sm font-medium text-neutral-100">
													{step.tool}
												</p>
												<p className="text-sm text-neutral-400">
													{step.description}
												</p>
											</div>
										</div>
									))}
								</div>
							</div>
						)}
						{run.progress_updates &&
							run.progress_updates.length > 0 && (
								<div>
									<h4 className="font-semibold text-neutral-300 mb-2">
										Execution Log
									</h4>
									<div className="bg-neutral-800/50 p-4 rounded-lg border border-neutral-700/50 space-y-4">
										{run.progress_updates.map(
											(update, index) => {
												const isLastUpdate =
													index ===
													run.progress_updates
														.length -
														1
												const isExecuting = [
													"processing",
													"planning"
												].includes(run.status)
												const messageContent =
													update.message?.content ||
													update.message
												const formattedTimestamp =
													new Date(
														update.timestamp
													).toLocaleTimeString([], {
														hour: "2-digit",
														minute: "2-digit",
														second: "2-digit"
													})

												if (
													isLastUpdate &&
													isExecuting &&
													update.message?.type ===
														"info" &&
													typeof messageContent ===
														"string"
												) {
													return (
														<TextShimmer
															key={index}
															className="font-mono text-sm text-brand-white"
															duration={2}
														>
															{messageContent}
														</TextShimmer>
													)
												}
												return (
													<ExecutionUpdate
														key={index}
														update={update}
													/>
												)
											}
										)}
									</div>
								</div>
							)}
						{run.result && (
							<TaskResultDisplay result={run.result} />
						)}
						{run.error && (
							<div>
								<h4 className="font-semibold text-neutral-300 mb-2">
									Error
								</h4>
								<p className="text-sm bg-red-500/10 border border-red-500/20 text-red-300 p-3 rounded-lg">
									{run.error}
								</p>
							</div>
						)}
					</div>
				))
			)}

			{(task.status === "completed" ||
				(task.chat_history && task.chat_history.length > 0)) && (
				<TaskChatSection
					task={task}
					onSendChatMessage={onSendChatMessage}
				/>
			)}
		</div>
	)
}

export default TaskDetailsContent
