import React, { useState, useEffect } from "react"
import ShiningButton from "@components/ShiningButton"
import ModalDialog from "@components/ModalDialog" // Assume this is available
import toast from "react-hot-toast"

const ModelSelection = ({ onProceed }) => {
	const [availableModels, setAvailableModels] = useState([])
	const [selectedModel, setSelectedModel] = useState("")
	const [apiKeyDialog, setApiKeyDialog] = useState(false)
	const [apiKey, setApiKey] = useState("")
	const [selectedProvider, setSelectedProvider] = useState("")

	const cloudModels = [
		{ name: "OpenAI", value: "openai" },
		{ name: "Gemini", value: "gemini" },
		{ name: "Claude", value: "claude" }
	]

	useEffect(() => {
		const fetchModels = async () => {
			try {
				const models = await window.electron.invoke("get-ollama-models")
				setAvailableModels(models)
				// Set a default model: first local model or first cloud model
				if (models.length > 0) {
					setSelectedModel(models[0])
					await saveModel(models[0])
				} else {
					setSelectedModel(cloudModels[0].value)
					await handleCloudModelSelection(cloudModels[0].value)
				}
			} catch (error) {
				toast.error("Error fetching available models.")
			}
		}
		fetchModels()
	}, [])

	const saveModel = async (model) => {
		try {
			await window.electron.invoke("set-db-data", {
				data: { selectedModel: model }
			})
		} catch (error) {
			toast.error("Error saving model selection.")
		}
	}

	const handleModelChange = async (model) => {
		setSelectedModel(model)
		if (cloudModels.some((m) => m.value === model)) {
			await handleCloudModelSelection(model)
		} else {
			await saveModel(model)
		}
	}

	const handleCloudModelSelection = async (model) => {
		setSelectedProvider(model)
		const hasKey = await window.electron.invoke("check-api-key", model)
		if (!hasKey) {
			setApiKeyDialog(true)
		} else {
			await saveModel(model)
		}
	}

	const handleApiKeySubmit = async () => {
		try {
			await window.electron.invoke("set-api-key", {
				provider: selectedProvider,
				apiKey
			})
			await saveModel(selectedProvider)
			setApiKeyDialog(false)
			setApiKey("")
			toast.success("API key saved and model selected.")
		} catch (error) {
			toast.error("Error saving API key.")
		}
	}

	return (
		<div className="w-4/5 max-w-lg font-Poppins bg-matteblack p-10 rounded-2xl shadow-2xl text-white">
			<h2 className="text-3xl font-bold mb-6 text-center">
				Select AI Model
			</h2>
			<p className="text-gray-300 mb-4 text-center">
				Choose a model to power your knowledge graph.
			</p>
			<select
				value={selectedModel}
				onChange={(e) => handleModelChange(e.target.value)}
				className="bg-transparent border-2 border-gray-400 text-white p-2 rounded-xl pr-4 w-full mb-4"
			>
				{cloudModels.map((model) => (
					<option
						className="bg-matteblack"
						key={model.value}
						value={model.value}
					>
						{model.name} (Cloud)
					</option>
				))}
				{availableModels.map((model) => (
					<option className="bg-matteblack" key={model} value={model}>
						{model} (Local)
					</option>
				))}
				{availableModels.length === 0 && (
					<option className="bg-matteblack" value="" disabled>
						No local models available.
					</option>
				)}
			</select>
			{apiKeyDialog && (
				<ModalDialog
					title={`Enter API Key for ${selectedProvider}`}
					description="Please enter your API key to use this model."
					onCancel={() => setApiKeyDialog(false)}
					onConfirm={handleApiKeySubmit}
					confirmButtonText="Save"
					showInput={true}
					inputValue={apiKey}
					onInputChange={setApiKey}
				/>
			)}
			<ShiningButton onClick={onProceed} className="w-full mt-4">
				Continue
			</ShiningButton>
		</div>
	)
}

export default ModelSelection
