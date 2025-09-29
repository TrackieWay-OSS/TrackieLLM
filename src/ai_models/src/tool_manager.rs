/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * TrackieLLM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TrackieLLM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with TrackieLLM. If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

//! Manages tool definitions and generates prompts and GBNF grammars for LLMs.

use super::ToolDefinition;
use serde_json::Value;

/// Manages a collection of tools available to the LLM.
#[derive(Debug, Default)]
pub struct ToolManager {
    tools: Vec<ToolDefinition>,
}

impl ToolManager {
    /// Creates a new, empty `ToolManager`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new tool.
    pub fn add_tool(&mut self, tool: ToolDefinition) {
        self.tools.push(tool);
    }

    /// Generates the "tools" section of the system prompt.
    /// This describes the available tools to the LLM.
    pub fn generate_tool_prompt_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut prompt = String::from(
            "You have access to the following tools. Respond with a JSON object in the format \\{\"name\": \"tool_name\", \"arguments\": {\\...\\}\\} to use a tool.\n\n"
        );
        prompt.push_str("Available tools:\n");

        for tool in &self.tools {
            let schema_str = serde_json::to_string_pretty(&tool.parameters_json_schema).unwrap_or_default();
            prompt.push_str(&format!(
                "- Name: {}\n  Description: {}\n  Argument Schema: {}\n",
                tool.name, tool.description, schema_str
            ));
        }

        prompt
    }

    /// Generates a GBNF grammar string to constrain the LLM's output
    /// to valid tool calls.
    pub fn generate_gbnf_grammar(&self) -> String {
        if self.tools.is_empty() {
            // If no tools, return a grammar that allows any character.
            return r#"root ::= [^]*"#.to_string();
        }

        let mut grammar = String::from("root ::= (tool_call)\n");
        let tool_names: Vec<String> = self
            .tools
            .iter()
            .map(|t| format!(r#""{}""#, t.name))
            .collect();

        let tool_name_rule = format!("tool_name ::= {}", tool_names.join(" | "));

        // This is a simplified grammar. A real-world scenario would need a more robust
        // JSON schema to GBNF converter.
        let arguments_rule = r#"
arguments ::= "{" (
  (ws string ws ":" ws value) ("," ws string ws ":" ws value)*
)? "}"
value ::= (string | number | "true" | "false" | "null" | object | array)
object ::= "{" ( (ws string ws ":" ws value) ("," ws string ws ":" ws value)* )? "}"
array ::= "[" ( (ws value) ("," ws value)* )? "]"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
number ::= ("-")? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?
ws ::= [ \t\n\r]*
"#;

        let tool_call_rule = format!(
            r#"
tool_call ::= "{{" ws "\"name\"" ws ":" ws tool_name ws "," ws "\"arguments\"" ws ":" ws arguments ws "}}"
{}"#,
            tool_name_rule
        );

        grammar.push_str(&tool_call_rule);
        grammar.push_str(arguments_rule);

        grammar
    }
}