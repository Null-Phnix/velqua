"""
Graph export for memory visualization.

Supports multiple graph formats:
- JSON graph (nodes + edges)
- GraphML (network visualization tools)
- DOT (Graphviz)
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set
from xml.dom import minidom

from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore


@dataclass
class GraphNode:
    """A node in the memory graph."""
    id: str
    node_type: str  # "episode", "fact", "topic"
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the memory graph."""
    source: str
    target: str
    edge_type: str  # "continues", "related", "source_of", "about"
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphData:
    """Complete graph data."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphExporter:
    """
    Exports memory relationships as graph data.

    Creates a graph where:
    - Episodes are nodes with temporal and topic info
    - Facts are nodes connected to source episodes
    - Topics create clustering connections
    - Continuation links create directed edges
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
    ):
        """Initialize exporter with stores."""
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store

    def build_graph(
        self,
        include_facts: bool = True,
        include_topics: bool = True,
        include_chunks: bool = True,
    ) -> GraphData:
        """
        Build graph data from memory stores.

        Args:
            include_facts: Include fact nodes and edges
            include_topics: Include topic clustering
            include_chunks: Include chunk relationships

        Returns:
            GraphData with nodes and edges
        """
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # Track what we've added
        node_ids: Set[str] = set()
        topics_seen: Dict[str, str] = {}  # topic -> topic_node_id

        # Get all episodes
        episodes = self.episodic_store.list_all(limit=10000)

        for ep in episodes:
            # Create episode node
            ep_node = GraphNode(
                id=ep.id,
                node_type="episode",
                label=ep.topic or ep.summary[:30] if ep.summary else "Episode",
                properties={
                    "summary": ep.summary,
                    "topic": ep.topic,
                    "importance": ep.importance,
                    "valence": ep.overall_valence.name if hasattr(ep.overall_valence, 'name') else str(ep.overall_valence),
                    "started_at": ep.started_at.isoformat() if ep.started_at else None,
                    "message_count": len(ep.messages),
                },
            )
            nodes.append(ep_node)
            node_ids.add(ep.id)

            # Topic clustering
            if include_topics and ep.topic:
                topic_key = ep.topic.lower().strip()
                if topic_key not in topics_seen:
                    topic_id = f"topic-{len(topics_seen)}"
                    topic_node = GraphNode(
                        id=topic_id,
                        node_type="topic",
                        label=ep.topic,
                        properties={"original_topic": ep.topic},
                    )
                    nodes.append(topic_node)
                    node_ids.add(topic_id)
                    topics_seen[topic_key] = topic_id

                # Edge from episode to topic
                edges.append(GraphEdge(
                    source=ep.id,
                    target=topics_seen[topic_key],
                    edge_type="about",
                    weight=0.5,
                ))

            # Continuation links from metadata
            if ep.metadata.get("continues_from"):
                edges.append(GraphEdge(
                    source=ep.metadata["continues_from"],
                    target=ep.id,
                    edge_type="continues",
                    weight=ep.metadata.get("continuation_confidence", 1.0),
                ))

            if ep.metadata.get("continues_to"):
                for target_id in ep.metadata["continues_to"]:
                    # Avoid duplicates (will be added from target's continues_from)
                    pass

            # Chunk relationships
            if include_chunks:
                if ep.metadata.get("previous_chunk"):
                    edges.append(GraphEdge(
                        source=ep.metadata["previous_chunk"],
                        target=ep.id,
                        edge_type="follows",
                        weight=1.0,
                    ))

        # Add facts if requested
        if include_facts:
            facts = self.semantic_store.list_all(limit=10000)

            for fact in facts:
                fact_node = GraphNode(
                    id=fact.id,
                    node_type="fact",
                    label=fact.content[:50] + "..." if len(fact.content) > 50 else fact.content,
                    properties={
                        "content": fact.content,
                        "fact_type": fact.fact_type,
                        "confidence": fact.confidence,
                        "importance": fact.importance,
                        "confirmation_count": fact.confirmation_count,
                    },
                )
                nodes.append(fact_node)
                node_ids.add(fact.id)

                # Connect to source episodes
                for ep_id in fact.source_episodes:
                    if ep_id in node_ids:
                        edges.append(GraphEdge(
                            source=ep_id,
                            target=fact.id,
                            edge_type="source_of",
                            weight=fact.confidence,
                        ))

        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "episode_count": len(episodes),
                "fact_count": len([n for n in nodes if n.node_type == "fact"]),
                "topic_count": len(topics_seen),
            },
        )

    def to_json(self, graph: GraphData) -> str:
        """Export graph to JSON format."""
        data = {
            "metadata": graph.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "label": n.label,
                    **n.properties,
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                    **e.properties,
                }
                for e in graph.edges
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_graphml(self, graph: GraphData) -> str:
        """Export graph to GraphML format."""
        # Create root element
        graphml = ET.Element("graphml")
        graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")

        # Define node attributes
        node_attrs = [
            ("type", "string"),
            ("label", "string"),
            ("importance", "double"),
            ("confidence", "double"),
        ]
        for name, dtype in node_attrs:
            key = ET.SubElement(graphml, "key")
            key.set("id", f"n_{name}")
            key.set("for", "node")
            key.set("attr.name", name)
            key.set("attr.type", dtype)

        # Define edge attributes
        edge_attrs = [
            ("type", "string"),
            ("weight", "double"),
        ]
        for name, dtype in edge_attrs:
            key = ET.SubElement(graphml, "key")
            key.set("id", f"e_{name}")
            key.set("for", "edge")
            key.set("attr.name", name)
            key.set("attr.type", dtype)

        # Create graph element
        g = ET.SubElement(graphml, "graph")
        g.set("id", "memory_graph")
        g.set("edgedefault", "directed")

        # Add nodes
        for node in graph.nodes:
            n = ET.SubElement(g, "node")
            n.set("id", node.id)

            # Add data elements
            d_type = ET.SubElement(n, "data")
            d_type.set("key", "n_type")
            d_type.text = node.node_type

            d_label = ET.SubElement(n, "data")
            d_label.set("key", "n_label")
            d_label.text = node.label

            if "importance" in node.properties:
                d_imp = ET.SubElement(n, "data")
                d_imp.set("key", "n_importance")
                d_imp.text = str(node.properties["importance"])

            if "confidence" in node.properties:
                d_conf = ET.SubElement(n, "data")
                d_conf.set("key", "n_confidence")
                d_conf.text = str(node.properties["confidence"])

        # Add edges
        for i, edge in enumerate(graph.edges):
            e = ET.SubElement(g, "edge")
            e.set("id", f"e{i}")
            e.set("source", edge.source)
            e.set("target", edge.target)

            d_type = ET.SubElement(e, "data")
            d_type.set("key", "e_type")
            d_type.text = edge.edge_type

            d_weight = ET.SubElement(e, "data")
            d_weight.set("key", "e_weight")
            d_weight.text = str(edge.weight)

        # Pretty print
        xml_string = ET.tostring(graphml, encoding="unicode")
        return minidom.parseString(xml_string).toprettyxml(indent="  ")

    def to_dot(self, graph: GraphData) -> str:
        """Export graph to DOT format (Graphviz)."""
        lines = ["digraph memory_graph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        lines.append("")

        # Node styles by type
        lines.append("  // Node definitions")
        for node in graph.nodes:
            # Escape label
            label = node.label.replace('"', '\\"').replace('\n', '\\n')[:40]

            # Style by type
            if node.node_type == "episode":
                style = 'shape=box,style=filled,fillcolor=lightblue'
            elif node.node_type == "fact":
                style = 'shape=ellipse,style=filled,fillcolor=lightyellow'
            elif node.node_type == "topic":
                style = 'shape=diamond,style=filled,fillcolor=lightgreen'
            else:
                style = 'shape=box'

            lines.append(f'  "{node.id}" [label="{label}",{style}];')

        lines.append("")
        lines.append("  // Edge definitions")

        for edge in graph.edges:
            # Edge style by type
            if edge.edge_type == "continues":
                style = 'style=bold,color=blue'
            elif edge.edge_type == "follows":
                style = 'style=dashed,color=blue'
            elif edge.edge_type == "source_of":
                style = 'style=dotted,color=gray'
            elif edge.edge_type == "about":
                style = 'style=dashed,color=green'
            else:
                style = ''

            lines.append(f'  "{edge.source}" -> "{edge.target}" [{style}];')

        lines.append("}")
        return "\n".join(lines)

    def get_stats(self, graph: GraphData) -> Dict[str, Any]:
        """Get statistics about the graph."""
        node_types = {}
        edge_types = {}

        for node in graph.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        for edge in graph.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "nodes_by_type": node_types,
            "edges_by_type": edge_types,
            "avg_edges_per_node": len(graph.edges) / len(graph.nodes) if graph.nodes else 0,
        }
