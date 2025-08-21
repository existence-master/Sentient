"use client"

import React, { useState } from "react"
import { format, parseISO } from "date-fns"
import { taskStatusColors, priorityMap } from "./constants"
import { cn } from "@utils/cn"
import CollapsibleSection from "./CollapsibleSection"
import ExecutionUpdate from "./ExecutionUpdate"
import ReactMarkdown from "react-markdown"
import toast from "react-hot-toast"
import { IconLoader } from "@tabler/icons-react"

// This component is used to handle clarification questions
// within recurring task runs. It's modified to use the run's status.
const QnaSection = ({ questions, task, onAnswerClarifications, runStatus }) => {
	const [answers, setAnswers] = useState({})
	const [isSubmitting, setIsSubmitting] = useState(false)
	// Use the status of the specific run to determine if input is needed.
	const isInputMode = runStatus === "clarification_pending"

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
		await onAnswerClarifications(task.task_id, answersPayload)
		setIsSubmitting(false)
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

const RecurringTaskDetails = ({ task, onAnswerClarifications }) => {
	if (!task) return null

	const statusInfo = taskStatusColors[task.status] || taskStatusColors.default
	const priorityInfo = priorityMap[task.priority] || priorityMap.default
	const schedule = task.schedule || {}
	const scheduleText =
		schedule.frequency === "daily"
			? `Recurring: Daily at ${schedule.time}`
			: `Recurring: Weekly on ${schedule.days?.join(", ") || "selected days"} at ${schedule.time}`

	return (
		<div className="space-y-6">
			{/* Meta Info */}
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Meta
				</label>
				<div className="flex items-center gap-4 text-sm bg-neutral-800/50 p-3 rounded-lg">
					<span className="text-sm text-neutral-400">Status:</span>
					<span
						className={cn(
							"font-semibold py-0.5 px-2 rounded-full text-xs flex items-center gap-1",
							statusInfo.color,
							statusInfo.border.replace("border-", "bg-") + "/20"
						)}
					>
						<statusInfo.icon size={12} />
						{statusInfo.label}
					</span>
					<div className="w-px h-4 bg-neutral-700"></div>
					<span className="text-sm text-neutral-400">Priority:</span>
					<span className={cn("font-semibold", priorityInfo.color)}>
						{priorityInfo.label}
					</span>
				</div>
			</div>

			{/* --- Researched Context (if planning) --- */}
			{task.status === "planning" && task.found_context && (
				<div>
					<label className="text-sm font-medium text-neutral-400 block mb-2">
						Researched Context
					</label>
					<div className="bg-neutral-800/50 p-3 rounded-lg text-sm text-neutral-300 whitespace-pre-wrap border border-neutral-700/50">
						<ReactMarkdown className="prose prose-sm prose-invert">
							{task.found_context}
						</ReactMarkdown>
					</div>
				</div>
			)}

			{/* Schedule */}
			<div>
				<label className="text-sm font-medium text-neutral-400 block mb-2">
					Schedule
				</label>
				<div className="bg-neutral-800/50 p-3 rounded-lg text-sm">
					{scheduleText}
				</div>
			</div>

			{/* Plan */}
			{task.plan && task.plan.length > 0 && (
				<div>
					<h4 className="font-semibold text-neutral-300 mb-2">
						Plan
					</h4>
					<div className="space-y-2">
						{task.plan.map((step, index) => (
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

			{/* Run History */}
			{task.runs && task.runs.length > 0 && (
				<div>
					<h4 className="font-semibold text-neutral-300 mb-2">
						Run History
					</h4>
					<div className="space-y-3">
						{task.runs
							.slice()
							.reverse()
							.map((run, index) => {
								const runNumber = task.runs.length - index
								const runStatusInfo =
									taskStatusColors[run.status] ||
									taskStatusColors.default
								const runDate = run.execution_start_time
									? parseISO(run.execution_start_time)
									: null
								const title = (
									<div className="flex flex-col gap-1 w-full">
										<div className="flex items-center justify-between w-full">
											<div className="flex items-center gap-2">
												<span className="font-semibold text-neutral-200 text-sm">
													Run #{runNumber}
												</span>
											</div>
											<span
												className={cn(
													"font-semibold text-xs flex items-center gap-1.5 py-0.5 px-2 rounded-full",
													runStatusInfo.color,
													runStatusInfo.border.replace(
														"border-",
														"bg-"
													) + "/20"
												)}
											>
												<runStatusInfo.icon size={12} />
												{runStatusInfo.label}
											</span>
										</div>
										<div className="text-xs text-neutral-400">
											{runDate
												? format(
														runDate,
														"MMMM d, yyyy 'at' p"
													)
												: "Run pending..."}
										</div>
									</div>
								)
								return (
									<CollapsibleSection
										key={run.run_id}
										title={title}
										defaultOpen={index === 0}
									>
										<div className="bg-neutral-800/50 p-4 rounded-lg border border-neutral-700/50 space-y-4 mt-2">
											{run.clarifying_questions &&
												run.clarifying_questions
													.length > 0 && (
													<QnaSection
														questions={
															run.clarifying_questions
														}
														task={task}
														onAnswerClarifications={
															onAnswerClarifications
														}
														runStatus={run.status}
													/>
												)}
											{run.plan &&
												run.plan.length > 0 && (
													<div>
														<h4 className="font-semibold text-neutral-300 mb-2">
															Plan
														</h4>
														<div className="space-y-2">
															{run.plan.map(
																(
																	step,
																	index
																) => (
																	<div
																		key={
																			index
																		}
																		className="flex items-start gap-3 p-3 bg-neutral-900/50 rounded-lg border border-neutral-700/50"
																	>
																		<div className="flex-shrink-0 w-5 h-5 bg-neutral-700 rounded-full flex items-center justify-center text-xs font-bold">
																			{index +
																				1}
																		</div>
																		<div>
																			<p className="text-sm font-medium text-neutral-100">
																				{
																					step.tool
																				}
																			</p>
																			<p className="text-sm text-neutral-400">
																				{
																					step.description
																				}
																			</p>
																		</div>
																	</div>
																)
															)}
														</div>
													</div>
												)}

											{run.progress_updates &&
											run.progress_updates.length > 0 ? (
												run.progress_updates.map(
													(update, index) => (
														<ExecutionUpdate
															key={index}
															update={update}
														/>
													)
												)
											) : (
												<p className="text-sm text-neutral-500">
													No execution log for this
													run.
												</p>
											)}
										</div>
									</CollapsibleSection>
								)
							})}
					</div>
				</div>
			)}
		</div>
	)
}

export default RecurringTaskDetails
