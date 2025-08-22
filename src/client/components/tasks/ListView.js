"use client"
import React, { useMemo } from "react" // prettier-ignore
import { AnimatePresence, motion } from "framer-motion"
import {
	IconSearch,
	IconAlignBoxLeftMiddleFilled,
	IconPlus
} from "@tabler/icons-react"
import { groupTasksByDate } from "@utils/taskUtils"
import TaskCardList from "./TaskCardList"
import CollapsibleSection from "./CollapsibleSection"

const ListView = ({
	oneTimeTasks,
	activeWorkflows,
    longFormTasks,
	onSelectTask,
	searchQuery,
	onSearchChange
}) => {
	const { today, tomorrow, future } = groupTasksByDate(oneTimeTasks)

	const containerVariants = {
		hidden: { opacity: 1 }, // Let the parent control opacity
		visible: {
			opacity: 1,
			transition: { staggerChildren: 0.07 }
		}
	}

	const sections = [
		{ title: "Today", tasks: today },
		{ title: "Tomorrow", tasks: tomorrow },
		{ title: "Future", tasks: future }
	]

	const upcomingTasksCount =
		activeWorkflows.length + today.length + tomorrow.length + future.length

	if (upcomingTasksCount === 0 && !searchQuery) {
		return (
			<div className="flex flex-col items-center justify-center h-full text-center text-neutral-500 p-8">
				<div className="max-w-lg">
					<h3 className="text-3xl font-semibold text-neutral-300 mb-2">
						Your Task List is Empty
					</h3>
					<p className="text-lg">
						Open the Task Composer and create a new task.
					</p>
					<p className="mb-4 text-lg">
						You can create one-time tasks, recurring workflows, or
						complex automations. Sentient will then generate a plan
						for your approval.
					</p>
					<IconAlignBoxLeftMiddleFilled
						className="mx-auto mt-4 text-brand-white"
						size={48}
					/>
				</div>
			</div>
		)
	}

	return (
		<div className="p-6 space-y-4 overflow-y-auto custom-scrollbar h-full bg-transparent rounded-xl">
			<div className="relative">
				<IconSearch
					className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500"
					size={20}
				/>
				<input
					type="text"
					value={searchQuery}
					onChange={(e) => onSearchChange(e.target.value)}
					placeholder="Search tasks..."
					className="w-full bg-neutral-900 border border-neutral-700 rounded-lg pl-10 pr-4 py-2 text-white placeholder-neutral-500 focus:ring-2 focus:ring-sentient-blue"
				/>
			</div>

			<AnimatePresence>
				<div className="divide-y divide-zinc-700">
                    {longFormTasks.length > 0 && (
                        <CollapsibleSection
                            key="long-form"
                            title={`Long-Form Tasks (${longFormTasks.length})`}
                            defaultOpen={true}
                        >
                            <motion.div
                                className="space-y-3 pt-2"
                                variants={containerVariants}
                                initial="hidden"
                                animate="visible"
                            >
                                {longFormTasks.map((task) => (
                                    <TaskCardList
                                        key={task.task_id}
                                        task={task}
                                        onSelectTask={onSelectTask}
                                    />
                                ))}
                            </motion.div>
                        </CollapsibleSection>
                    )}
					{activeWorkflows.length > 0 && (
						<CollapsibleSection
							key="workflows"
							title={`Active Workflows (${activeWorkflows.length})`}
							defaultOpen={true}
						>
							<motion.div
								className="space-y-3 pt-2"
								variants={containerVariants}
								initial="hidden"
								animate="visible"
							>
								{activeWorkflows.map((task) => (
									<TaskCardList
										key={task.task_id}
										task={task}
										onSelectTask={onSelectTask}
									/>
								))}
							</motion.div>
						</CollapsibleSection>
					)}

					{sections.map(
						(section) =>
							section.tasks.length > 0 && (
								<CollapsibleSection
									key={section.title}
									title={`${section.title} (${section.tasks.length})`}
									defaultOpen={true}
								>
									<motion.div
										className="space-y-3 pt-2"
										variants={containerVariants}
										initial="hidden"
										animate="visible"
									>
										{section.tasks.map((task) => (
											<TaskCardList
												key={task.instance_id}
												task={task}
												onSelectTask={onSelectTask}
											/>
										))}
									</motion.div>
								</CollapsibleSection>
							)
					)}
				</div>
			</AnimatePresence>
		</div>
	)
}

export default ListView
