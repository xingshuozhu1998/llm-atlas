<script setup lang="ts">
import { withBase } from 'vitepress'

interface Node {
  x: number
  y: number
  w: number
  h: number
  t: string
  s?: string
  link: string
  cls: string
  solid?: boolean
}

const panels = [
  { x: 228, y: 112, w: 392, h: 428, title: '后训练', cls: 'p-post' },
  { x: 860, y: 330, w: 240, h: 240, title: 'Agent', cls: 'p-agent' }
]

const nodes: Node[] = [
  // 左列：基座与两条分支
  { x: 24, y: 70, w: 160, h: 64, t: '模型架构', s: 'Transformer · MoE · 多模态', link: '/architecture/', cls: 'n-base' },
  { x: 24, y: 252, w: 160, h: 64, t: '基座模型', s: 'Qwen · DeepSeek · Llama', link: '/base-models/', cls: 'n-base' },
  { x: 24, y: 450, w: 160, h: 64, t: 'AIGC 扩散', s: 'DiT · Flow · 视频', link: '/aigc/', cls: 'n-base' },

  // 后训练
  { x: 250, y: 252, w: 150, h: 64, t: 'SFT', s: '数据 · 模板 · Packing', link: '/sft/', cls: 'n-post' },
  { x: 440, y: 152, w: 160, h: 64, t: 'DPO 系列', s: 'IPO · KTO · SimPO', link: '/dpo/', cls: 'n-post' },
  { x: 440, y: 252, w: 160, h: 64, t: 'PPO · GRPO', s: 'DAPO · GSPO · CISPO', link: '/rlhf/', cls: 'n-post' },
  { x: 440, y: 352, w: 160, h: 64, t: 'OPD / 蒸馏', s: '白盒 · 黑盒 · 自蒸馏', link: '/opd/', cls: 'n-post' },
  { x: 250, y: 440, w: 350, h: 56, t: 'LoRA 系列', s: 'QLoRA · DoRA · PiSSA —— 可叠加在以上任一阶段', link: '/lora/', cls: 'n-post' },

  // 右侧：对齐之后
  { x: 680, y: 252, w: 150, h: 64, t: '对齐模型', s: '可用的模型', link: '/post-training/', cls: 'n-hub', solid: true },
  { x: 880, y: 120, w: 200, h: 64, t: '推理优化与解码', s: 'KV Cache · 量化 · 投机', link: '/inference/', cls: 'n-serve' },
  { x: 880, y: 230, w: 200, h: 64, t: '评测', s: '基准 · judge · Rubric', link: '/eval/', cls: 'n-serve' },
  { x: 680, y: 400, w: 150, h: 64, t: '推理模型', s: 'RLVR · 测试时扩展', link: '/reasoning/', cls: 'n-serve' },

  // Agent 三层
  { x: 880, y: 372, w: 200, h: 52, t: 'Agent 训练', s: 'Tool Use · Agentic RL', link: '/agent/', cls: 'n-agent' },
  { x: 880, y: 440, w: 200, h: 52, t: 'Harness 工程', s: '执行循环 · 上下文 · 沙箱', link: '/harness/', cls: 'n-agent' },
  { x: 880, y: 508, w: 200, h: 52, t: 'Skills', s: '能力封装与自迭代', link: '/skills/', cls: 'n-agent' },

  { x: 680, y: 560, w: 150, h: 56, t: 'RSI 自我改进', link: '/rsi/', cls: 'n-rsi' }
]

const edges = [
  // 基座的两条分支
  { d: 'M 104 252 L 104 134', dashed: true, label: '架构', lx: 112, ly: 200, anchor: 'start' },
  { d: 'M 104 316 L 104 450', dashed: true, label: '另一条赛道', lx: 112, ly: 390, anchor: 'start' },
  // 基座 → SFT
  { d: 'M 184 284 L 250 284' },
  // SFT 分出三条训练目标
  { d: 'M 400 284 C 420 284 420 184 440 184' },
  { d: 'M 400 284 L 440 284' },
  { d: 'M 400 284 C 420 284 420 384 440 384' },
  // 汇到对齐模型
  { d: 'M 600 184 C 640 184 640 268 680 268' },
  { d: 'M 600 284 L 680 284' },
  { d: 'M 600 384 C 640 384 640 300 680 300' },
  // RL → 推理模型
  { d: 'M 600 300 C 648 318 652 388 680 426', label: '可验证奖励', lx: 656, ly: 350, anchor: 'start' },
  // 对齐模型 → 推理 / 评测 / Agent
  { d: 'M 830 274 C 856 274 856 152 880 152' },
  { d: 'M 830 284 C 856 284 856 262 880 262', label: '度量', lx: 855, ly: 240, anchor: 'middle' },
  { d: 'M 830 296 C 856 296 856 398 880 398' },
  // Agent 内部
  { d: 'M 980 424 L 980 440' },
  { d: 'M 980 492 L 980 508' },
  // Agent → RSI
  { d: 'M 880 534 C 852 534 848 588 830 588', dashed: true, label: '自我改进', lx: 838, ly: 548, anchor: 'end' }
]
</script>

<template>
  <figure class="atlas-flow">
    <div class="atlas-flow-scroll">
      <svg viewBox="0 0 1110 650" role="img" aria-label="LLM Atlas 全景图：从基座模型经后训练到推理、评测与 Agent">
        <defs>
          <marker id="af-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 9 5 L 0 9 z" />
          </marker>
        </defs>

        <!-- 分组底板 -->
        <g v-for="p in panels" :key="p.title" :class="p.cls">
          <rect class="panel" :x="p.x" :y="p.y" :width="p.w" :height="p.h" rx="10" />
          <text class="panel-title" :x="p.x + 18" :y="p.y + 30">{{ p.title }}</text>
        </g>

        <!-- 连线 -->
        <g class="edges">
          <path
            v-for="(e, i) in edges"
            :key="'e' + i"
            :d="e.d"
            :class="{ dashed: e.dashed }"
            marker-end="url(#af-arrow)"
          />
          <template v-for="(e, i) in edges" :key="'l' + i">
            <text v-if="e.label" class="edge-label" :x="e.lx" :y="e.ly" :text-anchor="e.anchor">{{ e.label }}</text>
          </template>
        </g>

        <!-- 节点 -->
        <a v-for="n in nodes" :key="n.t" :href="withBase(n.link)" :class="['node', n.cls, { solid: n.solid }]">
          <rect class="card" :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="8" />
          <rect class="bar" :x="n.x" :y="n.y" width="4" :height="n.h" />
          <text class="t" :x="n.x + 18" :y="n.s ? n.y + n.h / 2 - 2 : n.y + n.h / 2 + 6">{{ n.t }}</text>
          <text v-if="n.s" class="s" :x="n.x + 18" :y="n.y + n.h / 2 + 18">{{ n.s }}</text>
        </a>
      </svg>
    </div>
    <figcaption>从基座模型出发：后训练决定「学什么」（LoRA 决定「更新哪些参数」），对齐之后再往推理、评测、Agent 延伸。点任意方块进入对应章节。</figcaption>
  </figure>
</template>

<style>
.atlas-flow {
  --af-ink: var(--vp-c-text-1);
  --af-dim: var(--vp-c-text-2);
  --af-edge: #8a94a6;

  --af-base: #6b7280;
  --af-base-bg: #f2f4f6;
  --af-post: #1a4b8c;
  --af-post-bg: #e8f0fe;
  --af-post-panel: #f6f9fe;
  --af-serve: #2e7d4f;
  --af-serve-bg: #eaf5ec;
  --af-agent: #a9701f;
  --af-agent-bg: #fdf2e3;
  --af-agent-panel: #fffaf2;
  --af-rsi: #7c5cbf;
  --af-rsi-bg: #f2edfb;

  margin: 28px 0;
}

.dark .atlas-flow {
  --af-edge: #7c8798;
  --af-base: #aab3c0;
  --af-base-bg: #26292e;
  --af-post: #8ab4f8;
  --af-post-bg: #1d2735;
  --af-post-panel: #191d24;
  --af-serve: #6cc08b;
  --af-serve-bg: #1a2620;
  --af-agent: #e0a85c;
  --af-agent-bg: #2a2318;
  --af-agent-panel: #211c14;
  --af-rsi: #b49bea;
  --af-rsi-bg: #231d31;
}

.atlas-flow-scroll {
  overflow-x: auto;
}

.atlas-flow svg {
  display: block;
  width: 100%;
  min-width: 720px;
  height: auto;
  font-family: var(--vp-font-family-base);
}

/* 分组底板 */
.atlas-flow .panel {
  stroke-width: 1;
  stroke-dasharray: 4 4;
}
.atlas-flow .panel-title {
  font-size: 16.5px;
  font-weight: 650;
}
.atlas-flow .p-post .panel {
  fill: var(--af-post-panel);
  stroke: var(--af-post);
}
.atlas-flow .p-post .panel-title {
  fill: var(--af-post);
}
.atlas-flow .p-agent .panel {
  fill: var(--af-agent-panel);
  stroke: var(--af-agent);
}
.atlas-flow .p-agent .panel-title {
  fill: var(--af-agent);
}

/* 连线 */
.atlas-flow .edges path {
  fill: none;
  stroke: var(--af-edge);
  stroke-width: 1.6;
}
.atlas-flow .edges path.dashed {
  stroke-dasharray: 5 5;
}
.atlas-flow marker path {
  fill: var(--af-edge);
  stroke: none;
}
.atlas-flow .edge-label {
  font-size: 14px;
  fill: var(--af-dim);
}

/* 节点 */
.atlas-flow a.node {
  text-decoration: none;
  cursor: pointer;
}
.atlas-flow .card {
  stroke-width: 1;
  transition: filter 0.15s ease;
}
.atlas-flow .t {
  font-size: 17px;
  font-weight: 650;
  fill: var(--af-ink);
}
.atlas-flow .s {
  font-size: 13.5px;
  fill: var(--af-dim);
}

.atlas-flow .n-base .card { fill: var(--af-base-bg); stroke: var(--af-base); }
.atlas-flow .n-base .bar { fill: var(--af-base); }
.atlas-flow .n-post .card { fill: var(--af-post-bg); stroke: var(--af-post); }
.atlas-flow .n-post .bar { fill: var(--af-post); }
.atlas-flow .n-serve .card { fill: var(--af-serve-bg); stroke: var(--af-serve); }
.atlas-flow .n-serve .bar { fill: var(--af-serve); }
.atlas-flow .n-agent .card { fill: var(--af-agent-bg); stroke: var(--af-agent); }
.atlas-flow .n-agent .bar { fill: var(--af-agent); }
.atlas-flow .n-rsi .card { fill: var(--af-rsi-bg); stroke: var(--af-rsi); }
.atlas-flow .n-rsi .bar { fill: var(--af-rsi); }

/* 焦点节点：实心 */
.atlas-flow .n-hub .card { fill: var(--af-post); stroke: var(--af-post); }
.atlas-flow .n-hub .bar { fill: var(--af-post); }
.atlas-flow .n-hub .t { fill: #ffffff; }
.atlas-flow .n-hub .s { fill: rgba(255, 255, 255, 0.78); }
.dark .atlas-flow .n-hub .t { fill: #16181d; }
.dark .atlas-flow .n-hub .s { fill: rgba(22, 24, 29, 0.72); }

.atlas-flow a.node:hover .card {
  filter: brightness(0.96);
}
.dark .atlas-flow a.node:hover .card {
  filter: brightness(1.35);
}
.atlas-flow a.node:hover .bar {
  width: 8px;
}

.atlas-flow figcaption {
  margin-top: 10px;
  font-size: 14px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}
</style>
