import { repairJson } from '@toolsycc/json-repair';
import z from 'zod';
import { parse } from 'partial-json';
import BaseLLM from '../../base/llm';
import {
  GenerateObjectInput,
  GenerateTextInput,
  GenerateTextOutput,
  StreamTextOutput,
  ToolCall,
} from '../../types';
import { Message } from '@/lib/types';

type OpenCodeGoAnthropicConfig = {
  apiKey: string;
  model: string;
};

type AnthropicTextBlock = {
  type: 'text';
  text: string;
};

type AnthropicThinkingBlock = {
  type: 'thinking';
  thinking: string;
};

type AnthropicToolUseBlock = {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, any>;
};

type AnthropicBlock =
  | AnthropicTextBlock
  | AnthropicThinkingBlock
  | AnthropicToolUseBlock;

type AnthropicResponse = {
  content: AnthropicBlock[];
  stop_reason: string | null;
};

const OPEN_CODE_GO_MESSAGES_URL = 'https://opencode.ai/zen/go/v1/messages';
const ANTHROPIC_VERSION = '2023-06-01';

class OpenCodeGoAnthropicLLM extends BaseLLM<OpenCodeGoAnthropicConfig> {
  private extractSystem(messages: Message[]) {
    const system = messages
      .filter((message) => message.role === 'system')
      .map((message) => message.content)
      .join('\n\n')
      .trim();

    return system.length > 0 ? system : undefined;
  }

  private convertMessages(messages: Message[]) {
    return messages
      .filter((message) => message.role !== 'system')
      .map((message) => {
        if (message.role === 'user') {
          return {
            role: 'user',
            content: message.content,
          };
        }

        if (message.role === 'assistant') {
          if (message.tool_calls && message.tool_calls.length > 0) {
            return {
              role: 'assistant',
              content: message.tool_calls.map((toolCall) => ({
                type: 'tool_use',
                id: toolCall.id,
                name: toolCall.name,
                input: toolCall.arguments,
              })),
            };
          }

          return {
            role: 'assistant',
            content: message.content,
          };
        }

        return {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: message.id,
              content: message.content,
            },
          ],
        };
      });
  }

  private convertTools(input: GenerateTextInput) {
    return input.tools?.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: z.toJSONSchema(tool.schema),
    }));
  }

  private async createRequest(body: Record<string, any>) {
    const response = await fetch(OPEN_CODE_GO_MESSAGES_URL, {
      method: 'POST',
      headers: {
        'x-api-key': this.config.apiKey,
        'anthropic-version': ANTHROPIC_VERSION,
        'content-type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return response;
  }

  private getRequestBody(input: GenerateTextInput, stream: boolean) {
    return {
      model: this.config.model,
      max_tokens: input.options?.maxTokens ?? 2048,
      temperature: input.options?.temperature ?? 0,
      top_p: input.options?.topP,
      stop_sequences: input.options?.stopSequences,
      system: this.extractSystem(input.messages),
      messages: this.convertMessages(input.messages),
      tools: this.convertTools(input),
      stream,
    };
  }

  private extractText(content: AnthropicBlock[]) {
    return content
      .filter((block): block is AnthropicTextBlock => block.type === 'text')
      .map((block) => block.text)
      .join('');
  }

  private extractToolCalls(content: AnthropicBlock[]): ToolCall[] {
    return content
      .filter(
        (block): block is AnthropicToolUseBlock => block.type === 'tool_use',
      )
      .map((block) => ({
        id: block.id,
        name: block.name,
        arguments: block.input,
      }));
  }

  private parseJsonText<T>(text: string, schema: z.ZodType<T>) {
    const cleaned = text
      .trim()
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i, '')
      .replace(/\s*```$/, '')
      .trim();

    return schema.parse(JSON.parse(repairJson(cleaned, { extractJson: true }) as string));
  }

  async generateText(input: GenerateTextInput): Promise<GenerateTextOutput> {
    const response = await this.createRequest(this.getRequestBody(input, false));
    const data = (await response.json()) as AnthropicResponse;

    return {
      content: this.extractText(data.content),
      toolCalls: this.extractToolCalls(data.content),
      additionalInfo: {
        finishReason: data.stop_reason,
      },
    };
  }

  async *streamText(
    input: GenerateTextInput,
  ): AsyncGenerator<StreamTextOutput> {
    const response = await this.createRequest(this.getRequestBody(input, true));

    if (!response.body) {
      throw new Error('No stream body returned from OpenCode Go');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    const toolCalls = new Map<number, { id: string; name: string; arguments: string }>();

    const reader = response.body.getReader();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      while (buffer.includes('\n\n')) {
        const boundary = buffer.indexOf('\n\n');
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const dataLine = rawEvent
          .split('\n')
          .find((line) => line.startsWith('data: '));

        if (!dataLine) {
          continue;
        }

        const payload = JSON.parse(dataLine.slice(6));

        if (payload.type === 'content_block_start') {
          if (payload.content_block?.type === 'tool_use') {
            toolCalls.set(payload.index, {
              id: payload.content_block.id,
              name: payload.content_block.name,
              arguments:
                payload.content_block.input &&
                Object.keys(payload.content_block.input).length > 0
                  ? JSON.stringify(payload.content_block.input)
                  : '',
            });
          }

          continue;
        }

        if (payload.type === 'content_block_delta') {
          if (payload.delta?.type === 'text_delta') {
            yield {
              contentChunk: payload.delta.text || '',
              toolCallChunk: [],
              done: false,
            };
            continue;
          }

          if (payload.delta?.type === 'input_json_delta') {
            const current = toolCalls.get(payload.index);

            if (!current) {
              continue;
            }

            current.arguments += payload.delta.partial_json || '';

            let parsedArguments: Record<string, any> = {};

            try {
              parsedArguments = parse(current.arguments || '{}') as Record<string, any>;
            } catch {
              parsedArguments = {};
            }

            yield {
              contentChunk: '',
              toolCallChunk: [
                {
                  id: current.id,
                  name: current.name,
                  arguments: parsedArguments,
                },
              ],
              done: false,
            };
          }

          continue;
        }

        if (payload.type === 'message_delta') {
          yield {
            contentChunk: '',
            toolCallChunk: [],
            done: payload.delta?.stop_reason !== null,
            additionalInfo: {
              finishReason: payload.delta?.stop_reason,
            },
          };
        }
      }
    }
  }

  async generateObject<T>(input: GenerateObjectInput): Promise<z.infer<T>> {
    const response = await this.generateText(input);
    return this.parseJsonText(response.content, input.schema as z.ZodType<z.infer<T>>);
  }

  async *streamObject<T>(
    input: GenerateObjectInput,
  ): AsyncGenerator<Partial<z.infer<T>>> {
    let text = '';

    for await (const chunk of this.streamText(input)) {
      text += chunk.contentChunk;

      if (text.trim().length === 0) {
        continue;
      }

      try {
        yield this.parseJsonText(text, input.schema as z.ZodType<z.infer<T>>);
      } catch {
        yield {} as Partial<z.infer<T>>;
      }
    }
  }
}

export default OpenCodeGoAnthropicLLM;
