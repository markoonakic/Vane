import { UIConfigField } from '@/lib/config/types';
import BaseEmbedding from '../../base/embedding';
import BaseLLM from '../../base/llm';
import BaseModelProvider from '../../base/provider';
import { Model, ModelList, ProviderMetadata } from '../../types';
import OpenAILLM from '../openai/openaiLLM';
import OpenCodeGoAnthropicLLM from './opencodeGoAnthropicLLM';
import { getConfiguredModelProviderById } from '@/lib/config/serverRegistry';

interface OpenCodeGoConfig {
  apiKey: string;
}

const OPEN_CODE_GO_OPENAI_BASE_URL = 'https://opencode.ai/zen/go/v1';

const openAICompatibleModels = new Set(['glm-5', 'kimi-k2.5']);
const anthropicCompatibleModels = new Set(['minimax-m2.7', 'minimax-m2.5']);

const chatModels: Model[] = [
  {
    name: 'GLM-5',
    key: 'glm-5',
  },
  {
    name: 'Kimi K2.5',
    key: 'kimi-k2.5',
  },
  {
    name: 'MiniMax M2.7',
    key: 'minimax-m2.7',
  },
  {
    name: 'MiniMax M2.5',
    key: 'minimax-m2.5',
  },
];

const providerConfigFields: UIConfigField[] = [
  {
    type: 'password',
    name: 'API Key',
    key: 'apiKey',
    description: 'Your OpenCode Go API key',
    required: true,
    placeholder: 'OpenCode Go API Key',
    env: 'OPENCODE_GO_API_KEY',
    scope: 'server',
  },
];

class OpenCodeGoProvider extends BaseModelProvider<OpenCodeGoConfig> {
  async getDefaultModels(): Promise<ModelList> {
    return {
      chat: chatModels,
      embedding: [],
    };
  }

  async getModelList(): Promise<ModelList> {
    const defaultModels = await this.getDefaultModels();
    const configProvider = getConfiguredModelProviderById(this.id)!;

    return {
      embedding: [
        ...defaultModels.embedding,
        ...configProvider.embeddingModels,
      ],
      chat: [...defaultModels.chat, ...configProvider.chatModels],
    };
  }

  async loadChatModel(key: string): Promise<BaseLLM<any>> {
    const modelList = await this.getModelList();
    const exists = modelList.chat.find((m) => m.key === key);

    if (!exists) {
      throw new Error('Error Loading OpenCode Go Chat Model. Invalid Model Selected');
    }

    if (openAICompatibleModels.has(key)) {
      return new OpenAILLM({
        apiKey: this.config.apiKey,
        model: key,
        baseURL: OPEN_CODE_GO_OPENAI_BASE_URL,
      });
    }

    if (anthropicCompatibleModels.has(key)) {
      return new OpenCodeGoAnthropicLLM({
        apiKey: this.config.apiKey,
        model: key,
      });
    }

    throw new Error('Error Loading OpenCode Go Chat Model. Model is configured but unsupported by the provider implementation.');
  }

  async loadEmbeddingModel(_key: string): Promise<BaseEmbedding<any>> {
    throw new Error('OpenCode Go does not support embedding models. Configure a separate embeddings provider.');
  }

  static parseAndValidate(raw: any): OpenCodeGoConfig {
    if (!raw || typeof raw !== 'object') {
      throw new Error('Invalid config provided. Expected object');
    }

    if (!raw.apiKey) {
      throw new Error('Invalid config provided. API key must be provided');
    }

    return {
      apiKey: String(raw.apiKey),
    };
  }

  static getProviderConfigFields(): UIConfigField[] {
    return providerConfigFields;
  }

  static getProviderMetadata(): ProviderMetadata {
    return {
      key: 'opencode-go',
      name: 'OpenCode Go',
    };
  }
}

export default OpenCodeGoProvider;
