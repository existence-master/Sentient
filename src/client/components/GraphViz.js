import { useEffect, useRef } from "react"
import { DataSet, Network } from "vis-network/standalone"
import React from "react"

const GraphVisualization = ({ nodes, edges }) => {
	const networkRef = useRef() // Ref for the container div

	useEffect(() => {
		const container = networkRef.current
		if (!container) return // Exit if container isn't ready

		// Ensure data is arrays before processing
		const safeNodes = Array.isArray(nodes) ? nodes : []
		const safeEdges = Array.isArray(edges) ? edges : []

		// --- Data Processing (deduplication) ---
		const uniqueNodes = [
			...new Map(safeNodes.map((node) => [node.id, node])).values()
		]
		const uniqueEdges = [
			...new Map(
				safeEdges.map((edge) => [
					`${edge.from}-${edge.to}-${edge.label || edge.properties?.type || ""}`,
					edge
				])
			).values()
		] // Make edge key more unique

		// --- Vis Network Data ---
		const visNodes = new DataSet(
			uniqueNodes.map((node) => ({
				id: node.id,
				label:
					node.properties?.name ||
					node.label ||
					node.id.substring(0, 8), // Use part of ID if no label
				title: node.label
					? `Label: ${node.label}\nID: ${node.id}`
					: `ID: ${node.id}` // Tooltip
				// Properties can be added to 'title' or handled via events
			}))
		)

		const visEdges = new DataSet(
			uniqueEdges.map((edge) => ({
				id: edge.id, // Use edge ID if available
				from: edge.from,
				to: edge.to,
				label: edge.label || edge.properties?.type || "",
				title: `Type: ${edge.label || edge.properties?.type || "related"}`, // Tooltip
				arrows: "to",
				smooth: { type: "curvedCW", roundness: 0.1 } // Subtle curve
			}))
		)

		const data = { nodes: visNodes, edges: visEdges }

		// --- Vis Network Options ---
		// MODIFIED: Adjusted options for better dark theme integration and layout
		const options = {
			nodes: {
				shape: "dot", // Simple dot shape
				size: 18, // Slightly larger default size
				font: { size: 14, color: "#e5e7eb" }, // Lighter gray for text
				color: {
					border: "#0ea5e9", // lightblue border (Tailwind sky-500)
					background: "#0284c7", // darkblue background (Tailwind sky-600)
					highlight: { border: "#67e8f9", background: "#0369a1" }, // Lighter/darker blues on highlight
					hover: { border: "#38bdf8", background: "#075985" } // Lighter/darker blues on hover
				},
				borderWidth: 2
			},
			edges: {
				font: { align: "middle", size: 11, color: "#9ca3af" }, // Medium gray for edge labels
				color: {
					color: "#374151", // Darker gray for edges (neutral-700)
					highlight: "#0ea5e9", // lightblue on highlight
					hover: "#38bdf8", // Lighter blue on hover
					opacity: 0.8
				},
				arrows: { to: { enabled: true, scaleFactor: 0.7 } }, // Slightly smaller arrows
				smooth: {
					// Keep smooth edges
					enabled: true,
					type: "continuous", // Or 'dynamic' or 'curvedCW' etc.
					roundness: 0.1
				},
				width: 1, // Default edge width
				hoverWidth: 1.5 // Slightly thicker on hover
			},
			interaction: {
				hover: true,
				navigationButtons: false, // Hide default buttons (can add custom ones if needed)
				keyboard: true,
				tooltipDelay: 200, // Faster tooltip
				zoomView: true,
				dragView: true
			},
			physics: {
				enabled: true,
				solver: "forceAtlas2Based",
				forceAtlas2Based: {
					// Fine-tune layout physics
					gravitationalConstant: -35, // Adjust repulsion strength
					centralGravity: 0.01,
					springLength: 100,
					springConstant: 0.08,
					damping: 0.6
				},
				stabilization: { iterations: 150 }
			},
			layout: {
				improvedLayout: true // Use improved layout algorithm
			}
		}

		// --- Network Initialization ---
		const network = new Network(container, data, options)

		// Optional: Add event listeners here if needed (e.g., click, doubleClick)
		// network.on("click", function (params) { ... });

		// Cleanup function for the useEffect hook
		return () => {
			if (network) {
				network.destroy()
				console.log("GraphVisualization destroyed")
			}
		}
	}, [nodes, edges]) // Rerun effect if nodes or edges change

	// MODIFIED: Ensure container has a dark background explicitly if needed
	return <div ref={networkRef} className="w-full h-full bg-matteblack" /> // Container div
}

export default GraphVisualization
