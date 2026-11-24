#!/usr/bin/env node
'use strict';

const http = require('node:http');
const crypto = require('node:crypto');

const backend = new URL(process.env.MNN_BACKEND_URL || 'http://127.0.0.1:8001');
const listenHost = process.env.MNN_PROXY_HOST || '127.0.0.1';
const listenPort = Number(process.env.MNN_PROXY_PORT || 8000);

function bodyText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => typeof part?.text === 'string' ? part.text : '').join('');
  }
  return '';
}

function serializePreviousToolCalls(message) {
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const markup = calls.map((call) => {
    const fn = call.function || {};
    let args = {};
    try {
      args = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments || '{}') : (fn.arguments || {});
    } catch {
      args = {};
    }
    const parameters = Object.entries(args).map(([name, value]) => {
      const text = typeof value === 'string' ? value : JSON.stringify(value);
      return `<parameter=${name}>\n${text}\n</parameter>`;
    }).join('\n');
    return `<tool_call>\n<function=${fn.name || 'unknown'}>\n${parameters}\n</function>\n</tool_call>`;
  }).join('\n');
  const prefix = bodyText(message.content).trim();
  return [prefix, markup].filter(Boolean).join('\n\n');
}

function qwenToolsPrompt(tools) {
  const entries = tools.map((tool) => JSON.stringify(tool)).join('\n');
  return [
    '# Tools',
    '',
    'You have access to the following functions:',
    '',
    '<tools>',
    entries,
    '</tools>',
    '',
    'If you choose to call a function ONLY reply in the following format with NO suffix:',
    '',
    '<tool_call>',
    '<function=example_function_name>',
    '<parameter=example_parameter_1>',
    'value_1',
    '</parameter>',
    '<parameter=example_parameter_2>',
    'This is the value for the second parameter',
    'that can span',
    'multiple lines',
    '</parameter>',
    '</function>',
    '</tool_call>',
    '',
    '<IMPORTANT>',
    'Reminder:',
    '- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags',
    '- Required parameters MUST be specified',
    '- You may provide optional reasoning for a function call in natural language BEFORE the function call, but NOT after',
    '- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls',
    '</IMPORTANT>',
  ].join('\n');
}

function toMnnMessages(messages, tools, toolChoice) {
  const converted = (Array.isArray(messages) ? messages : []).map((message) => {
    const role = message.role === 'developer' ? 'system' : (message.role || 'user');
    let content = message.content;
    if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
      content = serializePreviousToolCalls(message);
    } else if (content === null || content === undefined) {
      content = '';
    }
    const { tool_calls, ...rest } = message;
    return { ...rest, role, content };
  });

  const usableTools = Array.isArray(tools) && tools.length && toolChoice !== 'none' ? tools : [];
  if (!usableTools.length) return converted;

  const toolPrompt = qwenToolsPrompt(usableTools);
  const systemIndex = converted.findIndex((message) => message.role === 'system');
  if (systemIndex >= 0) {
    converted[systemIndex] = {
      ...converted[systemIndex],
      content: `${toolPrompt}\n\n${bodyText(converted[systemIndex].content)}`,
    };
  } else {
    converted.unshift({ role: 'system', content: toolPrompt });
  }
  return converted;
}

function propertyValue(raw, property) {
  const type = property?.type;
  if (!type || type === 'string') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    if (type === 'integer' || type === 'number') {
      const number = Number(raw);
      return Number.isFinite(number) ? number : raw;
    }
    if (type === 'boolean' && /^(true|false)$/i.test(raw)) return raw.toLowerCase() === 'true';
    return raw;
  }
}

function parseQwenToolCalls(content, tools) {
  if (typeof content !== 'string' || !content.includes('<tool_call>')) return null;
  const toolMap = new Map();
  for (const tool of tools || []) {
    if (tool?.type === 'function' && tool.function?.name) toolMap.set(tool.function.name, tool.function);
  }

  const matcher = /<tool_call>\s*<function=([^>\s]+)>([\s\S]*?)<\/function>\s*<\/tool_call>/g;
  const calls = [];
  let match;
  let lastEnd = 0;
  let prefix = '';
  while ((match = matcher.exec(content)) !== null) {
    const fnName = match[1].trim();
    const schema = toolMap.get(fnName);
    if (!schema) return null;
    if (calls.length === 0) prefix = content.slice(0, match.index).trim();

    const args = {};
    const parameterMatcher = /<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/g;
    let parameter;
    while ((parameter = parameterMatcher.exec(match[2])) !== null) {
      let raw = parameter[2];
      if (raw.startsWith('\n')) raw = raw.slice(1);
      if (raw.endsWith('\n')) raw = raw.slice(0, -1);
      const name = parameter[1].trim();
      args[name] = propertyValue(raw, schema.parameters?.properties?.[name]);
    }
    const required = schema.parameters?.required || [];
    if (required.some((name) => !(name in args))) return null;

    calls.push({
      id: `call_${crypto.randomUUID().replaceAll('-', '')}`,
      type: 'function',
      function: { name: fnName, arguments: JSON.stringify(args) },
    });
    lastEnd = matcher.lastIndex;
  }
  if (!calls.length || content.slice(lastEnd).trim()) return null;
  return { prefix: prefix || null, calls };
}

function splitThinking(message, thinkingEnabled) {
  let content = bodyText(message.content);
  let reasoning = message.reasoning_content || '';
  const thinkingEnd = content.indexOf('</think>');
  if (thinkingEnd >= 0 && (thinkingEnabled || /^\s*<think>/.test(content))) {
    reasoning = content.slice(0, thinkingEnd).replace(/^\s*<think>\s*/, '').trim();
    content = content.slice(thinkingEnd + '</think>'.length).trimStart();
  }
  return { content, reasoning };
}

function backendRequest(path, payload) {
  return new Promise((resolve, reject) => {
    const target = new URL(path, backend);
    const body = payload === undefined ? null : Buffer.from(JSON.stringify(payload));
    const request = http.request({
      hostname: target.hostname,
      port: target.port || 80,
      path: `${target.pathname}${target.search}`,
      method: payload === undefined ? 'GET' : 'POST',
      headers: body ? { 'content-type': 'application/json', 'content-length': body.length } : {},
      timeout: Number(process.env.MNN_BACKEND_TIMEOUT || 300000),
    }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve({ status: response.statusCode || 502, body: Buffer.concat(chunks) }));
    });
    request.on('timeout', () => request.destroy(new Error('MNN backend request timed out')));
    request.on('error', reject);
    if (body) request.write(body);
    request.end();
  });
}

function chunk(id, model, delta, finishReason = null) {
  return {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

function writeEvent(response, value) {
  response.write(`data: ${JSON.stringify(value)}\n\n`);
}

function streamCompletion(response, id, model, message, finishReason) {
  response.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'access-control-allow-origin': '*',
  });
  writeEvent(response, chunk(id, model, { role: 'assistant' }));
  if (message.reasoning_content) {
    writeEvent(response, chunk(id, model, { reasoning_content: message.reasoning_content }));
  }
  if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
    if (message.content) writeEvent(response, chunk(id, model, { content: message.content }));
    message.tool_calls.forEach((call, index) => {
      writeEvent(response, chunk(id, model, {
        tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.function.name, arguments: '' } }],
      }));
      writeEvent(response, chunk(id, model, {
        tool_calls: [{ index, function: { arguments: call.function.arguments } }],
      }));
    });
  } else if (message.content) {
    writeEvent(response, chunk(id, model, { content: message.content }));
  }
  writeEvent(response, chunk(id, model, {}, finishReason));
  response.end('data: [DONE]\n\n');
}

function readRequestBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > 32 * 1024 * 1024) {
        reject(new Error('request body exceeds 32 MiB'));
        request.destroy();
      } else {
        chunks.push(chunk);
      }
    });
    request.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(new Error('invalid JSON request body'));
      }
    });
    request.on('error', reject);
  });
}

const server = http.createServer(async (request, response) => {
  const pathname = new URL(request.url || '/', 'http://localhost').pathname;
  response.setHeader('access-control-allow-origin', '*');
  response.setHeader('access-control-allow-headers', 'authorization, content-type');
  response.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');

  if (request.method === 'OPTIONS') {
    response.writeHead(204);
    response.end();
    return;
  }

  try {
    if (request.method === 'GET' && (pathname === '/v1/models' || pathname === '/models')) {
      const upstream = await backendRequest(pathname === '/models' ? '/v1/models' : pathname);
      response.writeHead(upstream.status, { 'content-type': 'application/json' });
      response.end(upstream.body);
      return;
    }

    if (request.method !== 'POST' || (pathname !== '/v1/chat/completions' && pathname !== '/chat/completions')) {
      response.writeHead(404, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ error: { message: 'Not found' } }));
      return;
    }

    const input = await readRequestBody(request);
    const tools = Array.isArray(input.tools) ? input.tools : [];
    const messages = toMnnMessages(input.messages, tools, input.tool_choice);
    const upstream = await backendRequest('/v1/chat/completions', {
      model: input.model,
      messages,
      stream: false,
      enable_thinking: input.enable_thinking === true,
    });
    if (upstream.status < 200 || upstream.status >= 300) {
      response.writeHead(upstream.status, { 'content-type': 'application/json' });
      response.end(upstream.body);
      return;
    }

    const completion = JSON.parse(upstream.body.toString('utf8'));
    const baseMessage = completion.choices?.[0]?.message || { role: 'assistant', content: '' };
    const { content, reasoning } = splitThinking(baseMessage, input.enable_thinking === true);
    const parsed = tools.length ? parseQwenToolCalls(content, tools) : null;
    const message = parsed
      ? { role: 'assistant', content: parsed.prefix, tool_calls: parsed.calls }
      : { role: 'assistant', content };
    if (reasoning) message.reasoning_content = reasoning;
    const finishReason = parsed ? 'tool_calls' : (completion.choices?.[0]?.finish_reason || 'stop');
    const id = completion.id || `chatcmpl_${crypto.randomUUID().replaceAll('-', '')}`;
    const model = input.model || completion.model || '';

    if (input.stream) {
      streamCompletion(response, id, model, message, finishReason);
      return;
    }

    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({
      id,
      object: 'chat.completion',
      created: completion.created || Math.floor(Date.now() / 1000),
      model,
      choices: [{ index: 0, message, finish_reason: finishReason }],
      usage: completion.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    }));
  } catch (error) {
    console.error('[pi-mnn-proxy]', error.message);
    if (!response.headersSent) response.writeHead(502, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ error: { message: error.message } }));
  }
});

server.listen(listenPort, listenHost, () => {
  console.log(`[pi-mnn-proxy] listening on http://${listenHost}:${listenPort}; MNN backend ${backend.origin}`);
});
