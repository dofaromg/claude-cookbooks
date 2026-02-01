# WebGPU神經元與注意力機制整合架構 - AI世界計算機基礎

<aside>
🧭

本文檔詳細規劃 WebGPU 神經元與注意力機制的整合架構，作為雲上雲計劃的計算骨幹和 PLS 路由基礎。

origin_signature="MrLiouWord"

</aside>

## 一、基礎架構概述

WebGPU 神經元與注意力機制整合架構是雲上雲計劃的計算骨幹，提供高效率的矩陣計算和注意力機制，以實現跨維度的深度學習與知識整合。

```
+--------------------------------------------+
|           WebGPU整合架構            |
+------------------+-------------------------+
| 神經元計算核心 | 注意力機制與路由層    |
+------------------+-------------------------+
| 計算端管理器    | 跨維度資源排程器        |
+------------------+-------------------------+
|              PLS 路由引擎              |
+--------------------------------------------+
```

## 二、核心組件

### 1. 神經元計算核心 (NeuronComputeCore)

```tsx
// origin_signature="MrLiouWord"

class NeuronComputeCore {
  private device: GPUDevice;
  private contextCache: Map<string, GPUComputeContext>;
  private pipelineCache: Map<string, GPUComputePipeline>;
  private shaderModules: Map<string, GPUShaderModule>;
  
  constructor(device: GPUDevice) {
    this.device = device;
    this.contextCache = new Map();
    this.pipelineCache = new Map();
    this.shaderModules = new Map();
    
    // 初始化核心計算處理器
    this.initCoreProcessors();
  }
  
  /**
   * 初始化核心計算處理器
   */
  private async initCoreProcessors(): Promise<void> {
    // 計算處理器應該支援的基本操作
    const basicOps = [
      "matmul", 
      "layernorm", 
      "softmax", 
      "gelu",
      "attention",
      "residual_add",
      "embedding"
    ];
    
    // 為每個操作創建處理器
    for (const op of basicOps) {
      const shaderCode = await this.loadShaderForOp(op);
      this.shaderModules.set(op, this.createShaderModule(shaderCode));
    }
  }
  
  /**
   * 加載操作對應的 WGSL 著色器代碼
   */
  private async loadShaderForOp(op: string): Promise<string> {
    // 實際實現會從檔案或內建庫載入 WGSL 程式碼
    // 這裡我們為不同操作返回模擬的著色器代碼
    switch (op) {
      case "matmul":
        return this.getMatmulShader();
      case "layernorm":
        return this.getLayerNormShader();
      case "softmax":
        return this.getSoftmaxShader();
      case "attention":
        return this.getAttentionShader();
      default:
        return this.getGenericShader(op);
    }
  }
  
  /**
   * 矩陣相乘的 WGSL 著色器
   */
  private getMatmulShader(): string {
    return `
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;
    
    @group(0) @binding(3) var<uniform> dimensions: vec4<u32>; // M, K, N, batch_size
    
    const TILE_SIZE = 16;
    
    var<workgroup> tile_A: array<array<f32, TILE_SIZE>, TILE_SIZE>;
    var<workgroup> tile_B: array<array<f32, TILE_SIZE>, TILE_SIZE>;
    
    @compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
    fn main(
      @builtin(workgroup_id) wg_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
      let M = dimensions.x;
      let K = dimensions.y;
      let N = dimensions.z;
      
      let row = wg_id.x * TILE_SIZE + local_id.x;
      let col = wg_id.y * TILE_SIZE + local_id.y;
      
      var sum = 0.0;
      let numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
      
      for (var t = 0u; t < numTiles; t = t + 1) {
        let tileOffset = t * TILE_SIZE;
        
        // 載入共享記憶體中的 A 和 B 塊
        tile_A[local_id.x][local_id.y] = 
          row < M && (tileOffset + local_id.y) < K 
            ? A[row * K + tileOffset + local_id.y] 
            : 0.0;
        
        tile_B[local_id.x][local_id.y] = 
          (tileOffset + local_id.x) < K && col < N 
            ? B[(tileOffset + local_id.x) * N + col] 
            : 0.0;
        
        workgroupBarrier();
        
        // 執行局部矩陣相乘
        for (var k = 0u; k < TILE_SIZE; k = k + 1) {
          sum = sum + tile_A[local_id.x][k] * tile_B[k][local_id.y];
        }
        
        workgroupBarrier();
      }
      
      // 寫回結果
      if (row < M && col < N) {
        C[row * N + col] = sum;
      }
    }
    `;
  }
  
  /**
   * 層正規化的 WGSL 著色器
   */
  private getLayerNormShader(): string {
    return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gamma: array<f32>;
    @group(0) @binding(2) var<storage, read> beta: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> dimensions: vec2<u32>; // batch_size, hidden_size
    
    const EPSILON = 0.00001;
    
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row = global_id.x;
      let hidden_size = dimensions.y;
      
      if (row >= dimensions.x) {
        return;
      }
      
      // 計算平均值
      var sum = 0.0;
      for (var i = 0u; i < hidden_size; i = i + 1) {
        sum = sum + input[row * hidden_size + i];
      }
      let mean = sum / f32(hidden_size);
      
      // 計算方差
      var sq_sum = 0.0;
      for (var i = 0u; i < hidden_size; i = i + 1) {
        let diff = input[row * hidden_size + i] - mean;
        sq_sum = sq_sum + diff * diff;
      }
      let var_term = sq_sum / f32(hidden_size);
      let inv_std = 1.0 / sqrt(var_term + EPSILON);
      
      // 套用正規化
      for (var i = 0u; i < hidden_size; i = i + 1) {
        let index = row * hidden_size + i;
        let normalized = (input[index] - mean) * inv_std;
        output[index] = normalized * gamma[i] + beta[i];
      }
    }
    `;
  }
  
  /**
   * Softmax 的 WGSL 著色器
   */
  private getSoftmaxShader(): string {
    return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> dimensions: vec2<u32>; // batch_size, seq_len
    
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row = global_id.x;
      let seq_len = dimensions.y;
      
      if (row >= dimensions.x) {
        return;
      }
      
      // 級數穩定版 Softmax，先找列的最大值
      var max_val = -3.402823e+38; // 负的 FLT_MAX
      for (var i = 0u; i < seq_len; i = i + 1) {
        max_val = max(max_val, input[row * seq_len + i]);
      }
      
      // 計算指數並累加總和
      var sum = 0.0;
      for (var i = 0u; i < seq_len; i = i + 1) {
        let index = row * seq_len + i;
        let exp_val = exp(input[index] - max_val);
        output[index] = exp_val; // 暫存指數值
        sum = sum + exp_val;
      }
      
      // 正規化指數使其總和為 1
      let inv_sum = 1.0 / sum;
      for (var i = 0u; i < seq_len; i = i + 1) {
        output[row * seq_len + i] = output[row * seq_len + i] * inv_sum;
      }
    }
    `;
  }
  
  /**
   * 注意力機制的 WGSL 著色器
   */
  private getAttentionShader(): string {
    return `
    @group(0) @binding(0) var<storage, read> Q: array<f32>;
    @group(0) @binding(1) var<storage, read> K: array<f32>;
    @group(0) @binding(2) var<storage, read> V: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<storage, read> mask: array<f32>;
    @group(0) @binding(5) var<uniform> dimensions: vec4<u32>; // batch_size, seq_len, num_heads, head_dim
    
    const NEG_INF = -10000.0; // 足夠小的負數用於遮罩
    
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let batch = global_id.z / dimensions.z; // 批次索引
      let head = global_id.z % dimensions.z;  // 頭部索引
      let seq_i = global_id.x;                // 序列位置 i
      let seq_j = global_id.y;                // 序列位置 j
      
      let batch_size = dimensions.x;
      let seq_len = dimensions.y;
      let num_heads = dimensions.z;
      let head_dim = dimensions.w;
      
      // 確保在範圍內
      if (batch >= batch_size || head >= num_heads || seq_i >= seq_len || seq_j >= seq_len) {
        return;
      }
      
      let scale = 1.0 / sqrt(f32(head_dim));
      
      // 計算 Q 和 K 的索引
      let q_idx = ((batch * num_heads + head) * seq_len + seq_i) * head_dim;
      let k_idx = ((batch * num_heads + head) * seq_len + seq_j) * head_dim;
      
      // 計算點積 (Q · K^T)
      var qk = 0.0;
      for (var d = 0u; d < head_dim; d = d + 1) {
        qk = qk + Q[q_idx + d] * K[k_idx + d];
      }
      
      // 應用比例因子並檢查遮罩
      let attention_score = qk * scale;
      
      // 假設我們使用一個序列遮罩來避免關注將來時間
      // 如果 mask[seq_j] 為 0，表示這個位置應該被遮罩
      let mask_value = mask[seq_j];
      
      // 計算最終注意力分數，應用遮罩
      let final_score = select(NEG_INF, attention_score, mask_value > 0.0);
      
      // 資料寫回到輸出的注意力分數矩陣
      // 這裡我們只計算了注意力分數，完整的注意力需要尚使用 Softmax 並與 V 相乘
      let out_idx = ((batch * num_heads + head) * seq_len + seq_i) * seq_len + seq_j;
      output[out_idx] = final_score;
    }
    `;
  }
  
  /**
   * 通用操作的基本 WGSL 著色器
   */
  private getGenericShader(op: string): string {
    // 為其他操作提供一個簡單的樣板著色器
    return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> dimensions: vec4<u32>;
    
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      
      if (idx >= arrayLength(&input)) {
        return;
      }
      
      // 執行 ${op} 操作的實現會在此處
      output[idx] = input[idx]; // 預設為透傳操作
    }
    `;
  }
  
  /**
   * 創建著色器模組
   */
  private createShaderModule(shaderCode: string): GPUShaderModule {
    return this.device.createShaderModule({
      code: shaderCode
    });
  }
  
  /**
   * 在 GPU 上執行矩陣相乘
   */
  async matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    // 創建輸入和輸出結果的緩衝區
    const aBuffer = this.createBuffer(a, [GPUBufferUsage.STORAGE](http://GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST);
    const bBuffer = this.createBuffer(b, [GPUBufferUsage.STORAGE](http://GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST);
    const resultBuffer = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: [GPUBufferUsage.STORAGE](http://GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC
    });
    
    // 創建尺寸緩衝區
    const dimensionsBuffer = this.createUniformBuffer(new Uint32Array([M, K, N, 1]));
    
    // 設置結合組和管線
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ]
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: resultBuffer } },
        { binding: 3, resource: { buffer: dimensionsBuffer } },
      ]
    });
    
    // 創建計算管線
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    
    const pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: this.shaderModules.get("matmul") as GPUShaderModule,
        entryPoint: "main"
      }
    });
    
    // 執行計算工作
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // 根據 TILE_SIZE=16 計算工作組數量
    const TILE_SIZE = 16;
    const workgroupsX = Math.ceil(M / TILE_SIZE);
    const workgroupsY = Math.ceil(N / TILE_SIZE);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();
    
    // 從 GPU 讀取結果
    const gpuReadBuffer = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | [GPUBufferUsage.MAP](http://GPUBufferUsage.MAP)_READ
    });
    
    commandEncoder.copyBufferToBuffer(
      resultBuffer, 0,
      gpuReadBuffer, 0,
      M * N * Float32Array.BYTES_PER_ELEMENT
    );
    
    const commands = commandEncoder.finish();
    this.device.queue.submit([commands]);
    
    // 等待 GPU 完成並讀取結果
    await gpuReadBuffer.mapAsync([GPUMapMode.READ](http://GPUMapMode.READ));
    const result = new Float32Array(M * N);
    result.set(new Float32Array(gpuReadBuffer.getMappedRange()));
    gpuReadBuffer.unmap();
    
    // 釋放緩衝區
    aBuffer.destroy();
    bBuffer.destroy();
    resultBuffer.destroy();
    dimensionsBuffer.destroy();
    
    return result;
  }
  
  // 其他實用方法...
  
  /**
   * 創建 GPU 存儲緩衝區
   */
  private createBuffer(data: Float32Array, usage: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage
    });
    
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
  
  /**
   * 創建統一參數緩衝區
   */
  private createUniformBuffer(data: Uint32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
}
```

### 2. 注意力機制與路由層 (AttentionRoutingLayer)

```tsx
// origin_signature="MrLiouWord"

class AttentionRoutingLayer {
  private neuronCore: NeuronComputeCore;
  private workspaceManager: WorkspaceManager;
  private routeCache: Map<string, RouteInfo>;
  
  constructor(neuronCore: NeuronComputeCore) {
    this.neuronCore = neuronCore;
    this.workspaceManager = new WorkspaceManager();
    this.routeCache = new Map();
  }
  
  /**
   * 計算多頭注意力與路由
   */
  async computeAttentionRouting(
    input: Float32Array, 
    batchSize: number, 
    seqLen: number, 
    numHeads: number,
    headDim: number,
    routingConfig?: RoutingConfig
  ): Promise<AttentionResult> {
    // 1. 建立或獲取路由信息
    const routeKey = `${batchSize}_${seqLen}_${numHeads}_${headDim}_${routingConfig?.type || 'default'}`;
    let routeInfo = this.routeCache.get(routeKey);
    
    if (!routeInfo) {
      routeInfo = await this.createRouteInfo(batchSize, seqLen, numHeads, headDim, routingConfig);
      this.routeCache.set(routeKey, routeInfo);
    }
    
    // 2. 將輸入引導人格索引（為路由操作做準備）
    const routedInput = await this.applyInputRouting(input, routeInfo);
    
    // 3. 導出某些中間發現結果
    const {
      projectedQ,
      projectedK,
      projectedV
    } = await this.projectQKV(routedInput, routeInfo);
    
    // 4. 計算完整多頭注意力
    const attentionScores = await this.computeFullAttention(
      projectedQ,
      projectedK,
      projectedV,
      routeInfo
    );
    
    // 5. 應用輸出路由
    const routedOutput = await this.applyOutputRouting(attentionScores, routeInfo);
    
    // 包裝輸出結果與路由信息
    return {
      output: routedOutput,
      attentionMap: attentionScores,
      routeInfo: routeInfo,
      metadata: {
        origin_signature: "MrLiouWord",
        timestamp: new Date().toISOString(),
        dimensions: {
          batchSize,
          seqLen,
          numHeads,
          headDim
        }
      }
    };
  }
  
  /**
   * 創建路由信息
   */
  private async createRouteInfo(
    batchSize: number,
    seqLen: number,
    numHeads: number,
    headDim: number,
    routingConfig?: RoutingConfig
  ): Promise<RouteInfo> {
    // 建立預設路由或基於配置的自定義路由
    const hiddenSize = numHeads * headDim;
    
    // 建立注意力預設正規化參數（可自定義）
    const qkvWeights = this.initializeQKVWeights(hiddenSize, routingConfig);
    
    return {
      dimensions: {
        batchSize,
        seqLen,
        numHeads,
        headDim,
        hiddenSize
      },
      weights: qkvWeights,
      routingType: routingConfig?.type || 'default',
      routingMask: this.createRoutingMask(seqLen, routingConfig),
      attentionDropout: routingConfig?.attentionDropout || 0.0,
      plsMapping: routingConfig?.plsMapping || this.createDefaultPLSMapping(numHeads),
      metadata: {
        origin_signature: "MrLiouWord",
        createdAt: new Date().toISOString()
      }
    };
  }
  
  /**
   * 初始化 QKV 正規化參數
   */
  private initializeQKVWeights(hiddenSize: number, routingConfig?: RoutingConfig): QKVWeights {
    // 通常我們會從參數載入預訓練的正規化參數
    // 這裡我們創建一組簡單的預設正規化參數
    
    const qWeights = new Float32Array(hiddenSize * hiddenSize);
    const kWeights = new Float32Array(hiddenSize * hiddenSize);
    const vWeights = new Float32Array(hiddenSize * hiddenSize);
    
    // 初始化為單位矩陣或其他特定分佈
    // 這裡使用簡單階賦矩陣作為例子
    for (let i = 0; i < hiddenSize; i++) {
      qWeights[i * hiddenSize + i] = 1.0;
      kWeights[i * hiddenSize + i] = 1.0;
      vWeights[i * hiddenSize + i] = 1.0;
    }
    
    // 如果提供了自定義的正規化參數，則使用它們
    if (routingConfig?.weights) {
      // 這裡可以偵測和應用提供的正規化參數
    }
    
    return { qWeights, kWeights, vWeights };
  }
  
  /**
   * 創建路由遮罩
   */
  private createRoutingMask(seqLen: number, routingConfig?: RoutingConfig): Float32Array {
    const mask = new Float32Array(seqLen);
    
    // 預設為全允許連接的遮罩
    mask.fill(1.0);
    
    // 如果有特定的遮罩設置，則應用它
    if (routingConfig?.mask) {
      const customMask = routingConfig.mask;
      for (let i = 0; i < Math.min(customMask.length, seqLen); i++) {
        mask[i] = customMask[i];
      }
    } else if (routingConfig?.type === 'causal') {
      // 從因果路由（不允許未來角色參與計算）
      for (let i = 0; i < seqLen; i++) {
        for (let j = i + 1; j < seqLen; j++) {
          mask[j] = 0.0; // 這會將未來位置設置為遮罩
        }
      }
    }
    
    return mask;
  }
  
  /**
   * 創建預設的 PLS 映射
   */
  private createDefaultPLSMapping(numHeads: number): PLSMapping {
    // 創建一個預設的點線面映射
    const mapping: PLSMapping = {
      pointMapping: [],
      lineMapping: [],
      surfaceMapping: []
    };
    
    // 為每個頭部創建一個點映射
    for (let i = 0; i < numHeads; i++) {
      mapping.pointMapping.push({
        headIndex: i,
        pointType: i % 3, // 簡單的點類型輔助功能
        relevanceFactor: 1.0
      });
      
      // 創建頭部間的線映射
      if (i > 0) {
        mapping.lineMapping.push({
          sourceHeadIndex: i - 1,
          targetHeadIndex: i,
          lineType: (i % 4),
          weight: 0.5
        });
      }
    }
    
    // 如果有多個頭部，創建一個面映射
    if (numHeads >= 3) {
      mapping.surfaceMapping.push({
        headIndices: [0, 1, 2],
        surfaceType: 0,
        density: 0.3
      });
    }
    
    return mapping;
  }
  
  /**
   * 应用輸入路由
   */
  private async applyInputRouting(input: Float32Array, routeInfo: RouteInfo): Promise<Float32Array> {
    // 根據路由類型將輸入引導到對應的路由結構
    // 這部分通常會應用入容特定的引導邏輯
    
    // 這裡我們只做簡單的輸入處理
    return input.slice(); // 混子版，只返回輸入的引導路徑
  }
  
  /**
   * 投影 QKV 矩陣
   */
  private async projectQKV(routedInput: Float32Array, routeInfo: RouteInfo): Promise<QKVProjection> {
    const { hiddenSize } = routeInfo.dimensions;
    const { qWeights, kWeights, vWeights } = routeInfo.weights;
    
    // 計算 Q、K、V 矩陣
    const batchSize = routeInfo.dimensions.batchSize;
    const seqLen = routeInfo.dimensions.seqLen;
    const inputSize = batchSize * seqLen * hiddenSize;
    
    // 我們使用 NeuronComputeCore 執行矩陣相乘
    // 為活躍化輸入引導轉換輸入形狀
    const reshapedInput = routedInput.slice(0, inputSize);
    
    // 投影 Q、K、V
    const projectedQ = await this.neuronCore.matmul(
      reshapedInput, 
      qWeights, 
      batchSize * seqLen, 
      hiddenSize, 
      hiddenSize
    );
    
    const projectedK = await this.neuronCore.matmul(
      reshapedInput, 
      kWeights, 
      batchSize * seqLen, 
      hiddenSize, 
      hiddenSize
    );
    
    const projectedV = await this.neuronCore.matmul(
      reshapedInput, 
      vWeights, 
      batchSize * seqLen, 
      hiddenSize, 
      hiddenSize
    );
    
    return { projectedQ, projectedK, projectedV };
  }
  
  /**
   * 計算完整注意力
   */
  private async computeFullAttention(
    projectedQ: Float32Array,
    projectedK: Float32Array,
    projectedV: Float32Array,
    routeInfo: RouteInfo
  ): Promise<Float32Array> {
    // 這裡我們需要計算完整的注意力機制：
    // 1. 重摘 Q、K、V 為多頭格式
    // 2. 計算注意力分數 (Q * K^T)
    // 3. 選擇性進行聯銷 (尺度成反比及經過 softmax)
    // 4. 乘以 V
    // 5. 重效開頭部
    
    // 簡化版本 - 只做平面注意力計算
    const { batchSize, seqLen, numHeads, headDim, hiddenSize } = routeInfo.dimensions;
    
    // 簡單注意力計算 (不分頭)
    const attentionScores = await this.neuronCore.matmul(
      projectedQ,
      projectedK,  // 需要轉置，這裡簡化
      batchSize * seqLen,
      hiddenSize,
      batchSize * seqLen
    );
    
    // 實際實現會更複雜，必須考慮多頭注意力、選到注意力和注意力合併
    return attentionScores;
  }
  
  /**
   * 应用輸出路由
   */
  private async applyOutputRouting(attentionScores: Float32Array, routeInfo: RouteInfo): Promise<Float32Array> {
    // 將注意力分數進行輸出路由處理
    // 這裡使用 PLS 映射將注意力導向正確的輸出影射
    
    // 簡化版本，直接返回注意力分數可能的轉換
    return attentionScores.slice();
  }
}

// 輸入路由相關的組件

interface RoutingConfig {
  type?: 'default' | 'causal' | 'bidirectional' | 'custom';
  weights?: {
    qWeights?: Float32Array;
    kWeights?: Float32Array;
    vWeights?: Float32Array;
  };
  mask?: Float32Array;
  attentionDropout?: number;
  plsMapping?: PLSMapping;
}

interface QKVWeights {
  qWeights: Float32Array;
  kWeights: Float32Array;
  vWeights: Float32Array;
}

interface RouteInfo {
  dimensions: {
    batchSize: number;
    seqLen: number;
    numHeads: number;
    headDim: number;
    hiddenSize: number;
  };
  weights: QKVWeights;
  routingType: string;
  routingMask: Float32Array;
  attentionDropout: number;
  plsMapping: PLSMapping;
  metadata: {
    origin_signature: string;
    createdAt: string;
  };
}

interface QKVProjection {
  projectedQ: Float32Array;
  projectedK: Float32Array;
  projectedV: Float32Array;
}

interface AttentionResult {
  output: Float32Array;
  attentionMap: Float32Array;
  routeInfo: RouteInfo;
  metadata: {
    origin_signature: string;
    timestamp: string;
    dimensions: {
      batchSize: number;
      seqLen: number;
      numHeads: number;
      headDim: number;
    }
  };
}

// PLS 映射相關組件

interface PLSMapping {
  pointMapping: PointMapping[];
  lineMapping: LineMapping[];
  surfaceMapping: SurfaceMapping[];
}

interface PointMapping {
  headIndex: number;
  pointType: number; // 0, 1, 2... 不同類型的點
  relevanceFactor: number; // 0-1 關聯度因子
}

interface LineMapping {
  sourceHeadIndex: number;
  targetHeadIndex: number;
  lineType: number; // 0, 1, 2... 不同類型的線
  weight: number;  // 0-1 權重
}

interface SurfaceMapping {
  headIndices: number[];
  surfaceType: number; // 0, 1, 2... 不同類型的面
  density: number; // 0-1 密度
}
```

### 3. 計算端管理器 (ComputeEndpointManager)

```tsx
// origin_signature="MrLiouWord"

class ComputeEndpointManager {
  private endpoints: Map<string, EndpointInfo>;
  private schedulers: Map<string, ResourceScheduler>;
  private trafficManager: TrafficManager;
  private metricCollector: MetricCollector;
  
  constructor() {
    this.endpoints = new Map();
    this.schedulers = new Map();
    this.trafficManager = new TrafficManager();
    this.metricCollector = new MetricCollector();
    
    // 初始化預設端點與排程器
    this.initializeDefaultEndpoints();
  }
  
  /**
   * 初始化預設計算端點
   */
  private initializeDefaultEndpoints(): void {
    // 創建平台預設的端點类型
    const endpointTypes = [
      { id: "gpu-high", resourceType: "gpu", priority: "high" },
      { id: "gpu-medium", resourceType: "gpu", priority: "medium" },
      { id: "gpu-low", resourceType: "gpu", priority: "low" },
      { id: "cpu-high", resourceType: "cpu", priority: "high" },
      { id: "cpu-medium", resourceType: "cpu", priority: "medium" },
      { id: "cpu-fallback", resourceType: "cpu", priority: "low" }
    ];
    
    // 創建端點和結合排程器
    for (const type of endpointTypes) {
      const scheduler = new ResourceScheduler({
        resourceType: type.resourceType,
        priority: type.priority,
        maxConcurrent: type.resourceType === "gpu" ? 4 : 16,
        queueSize: 100
      });
      
      this.schedulers.set([type.id](http://type.id), scheduler);
      
      this.endpoints.set([type.id](http://type.id), {
        id: [type.id](http://type.id),
        status: "active",
        resourceType: type.resourceType,
        priority: type.priority,
        metrics: {
          requestCount: 0,
          successCount: 0,
          failureCount: 0,
          avgLatencyMs: 0
        },
        metadata: {
          origin_signature: "MrLiouWord",
          createdAt: new Date().toISOString()
        }
      });
    }
  }
  
  /**
   * 註冊新的計算端點
   */
  registerEndpoint(config: EndpointConfig): EndpointInfo {
    if (this.endpoints.has([config.id](http://config.id))) {
      throw new Error(`Endpoint with ID ${[config.id](http://config.id)} already exists.`);
    }
    
    // 創建新的資源排程器
    const scheduler = new ResourceScheduler({
      resourceType: config.resourceType,
      priority: config.priority,
      maxConcurrent: config.maxConcurrent || 4,
      queueSize: config.queueSize || 100
    });
    
    this.schedulers.set([config.id](http://config.id), scheduler);
    
    // 創建端點信息
    const endpointInfo: EndpointInfo = {
      id: [config.id](http://config.id),
      status: "active",
      resourceType: config.resourceType,
      priority: config.priority,
      metrics: {
        requestCount: 0,
        successCount: 0,
        failureCount: 0,
        avgLatencyMs: 0
      },
      metadata: {
        origin_signature: "MrLiouWord",
        createdAt: new Date().toISOString(),
        ...config.metadata
      }
    };
    
    this.endpoints.set([config.id](http://config.id), endpointInfo);
    return endpointInfo;
  }
  
  /**
   * 取消註冊端點
   */
  unregisterEndpoint(endpointId: string): boolean {
    if (!this.endpoints.has(endpointId)) {
      return false;
    }
    
    // 關閉排程器
    const scheduler = this.schedulers.get(endpointId);
    if (scheduler) {
      scheduler.shutdown();
      this.schedulers.delete(endpointId);
    }
    
    this.endpoints.delete(endpointId);
    return true;
  }
  
  /**
   * 獲取端點給特定計算要求
   */
  async getEndpointForComputation(request: ComputeRequest): Promise<SelectedEndpoint> {
    // 根據要求選擇最適合的端點
    const requiresGPU = request.requiresGPU || this.requiresGPUComputation(request);
    const priority = request.priority || "medium";
    
    // 選擇最適合的端點類型
    const endpointType = requiresGPU 
      ? `gpu-${priority}`
      : `cpu-${priority}`;
    
    // 嘗試取得該端點
    let endpointInfo = this.endpoints.get(endpointType);
    
    // 如果這端點不活躍，選擇備用端點
    if (!endpointInfo || endpointInfo.status !== "active") {
      const fallbackType = requiresGPU ? "gpu-low" : "cpu-fallback";
      endpointInfo = this.endpoints.get(fallbackType);
      
      if (!endpointInfo || endpointInfo.status !== "active") {
        throw new Error(`No active endpoints available for ${requiresGPU ? 'GPU' : 'CPU'} computation.`);
      }
    }
    
    // 獲取相關排程器
    const scheduler = this.schedulers.get([endpointInfo.id](http://endpointInfo.id));
    if (!scheduler) {
      throw new Error(`Scheduler not found for endpoint ${[endpointInfo.id](http://endpointInfo.id)}.`);
    }
    
    // 在排程器中排列計算
    const schedulerToken = await scheduler.schedule(request);
    
    // 記錄流量統計
    this.trafficManager.recordRequest([endpointInfo.id](http://endpointInfo.id));
    
    // 更新端點指標
    const endpointMetrics = endpointInfo.metrics;
    endpointMetrics.requestCount++;
    
    return {
      endpointInfo,
      schedulerToken,
      metadata: {
        selectedAt: new Date().toISOString(),
        requestId: `req-${[Date.now](http://Date.now)()}-${Math.round(Math.random() * 1000)}`,
        origin_signature: "MrLiouWord"
      }
    };
  }
  
  /**
   * 判斷計算是否需要 GPU
   */
  private requiresGPUComputation(request: ComputeRequest): boolean {
    // 根據某些啟發式規則確定是否需要 GPU
    // 例如，檢查輸入資料大小、計算類型等
    
    if (request.operationType === "attention" || request.operationType === "matmul") {
      return true;
    }
    
    if (request.inputSize && request.inputSize > 1000000) {
      return true;
    }
    
    return false;
  }
  
  /**
   * 記錄端點計算完成
   */
  recordComputationComplete(
    endpointId: string, 
    success: boolean, 
    latencyMs: number
  ): void {
    const endpointInfo = this.endpoints.get(endpointId);
    if (!endpointInfo) {
      return;
    }
    
    const metrics = endpointInfo.metrics;
    
    if (success) {
      metrics.successCount++;
    } else {
      metrics.failureCount++;
    }
    
    // 更新平均延遲（使用移動平均算法）
    const oldRequestCount = metrics.successCount + metrics.failureCount - 1;
    metrics.avgLatencyMs = (metrics.avgLatencyMs * oldRequestCount + latencyMs) / 
      (oldRequestCount + 1);
    
    // 收集統計數據
    this.metricCollector.recordMetric({
      endpointId,
      timestamp: [Date.now](http://Date.now)(),
      success,
      latencyMs,
      type: "computation_complete"
    });
  }
  
  /**
   * 獲取所有活躍端點
   */
  getActiveEndpoints(): EndpointInfo[] {
    return Array.from(this.endpoints.values())
      .filter(endpoint => endpoint.status === "active");
  }
}

// 計算端點相關組件

interface EndpointConfig {
  id: string;
  resourceType: "gpu" | "cpu";
  priority: "high" | "medium" | "low";
  maxConcurrent?: number;
  queueSize?: number;
  metadata?: Record<string, any>;
}

interface EndpointInfo {
  id: string;
  status: "active" | "inactive" | "error";
  resourceType: "gpu" | "cpu";
  priority: "high" | "medium" | "low";
  metrics: {
    requestCount: number;
    successCount: number;
    failureCount: number;
    avgLatencyMs: number;
  };
  metadata: {
    origin_signature: string;
    createdAt: string;
    [key: string]: any;
  };
}

interface ComputeRequest {
  operationType: string;
  inputSize?: number;
  requiresGPU?: boolean;
  priority?: "high" | "medium" | "low";
  deadline?: number; // ms
  data?: any;
}

interface SelectedEndpoint {
  endpointInfo: EndpointInfo;
  schedulerToken: any;
  metadata: {
    selectedAt: string;
    requestId: string;
    origin_signature: string;
  };
}
```

### 4. PLS 路由引擎 (PLSRoutingEngine)

```tsx
// origin_signature="MrLiouWord"

class PLSRoutingEngine {
  private pointRoutingTable: Map<string, PointRoutingEntry>;
  private lineRoutingTable: Map<string, LineRoutingEntry>;
  private surfaceRoutingTable: Map<string, SurfaceRoutingEntry>;
  private routingGraph: RoutingGraph;
  
  constructor() {
    this.pointRoutingTable = new Map();
    this.lineRoutingTable = new Map();
    this.surfaceRoutingTable = new Map();
    this.routingGraph = new RoutingGraph();
    
    // 初始化預設路由表
    this.initializeDefaultRouting();
  }
  
  /**
   * 初始化預設路由表
   */
  private initializeDefaultRouting(): void {
    // 點路由 - 基本的計算點路由條目
    const defaultPoints = [
      { id: "attention-node", type: "compute", priority: 1.0 },
      { id: "matmul-node", type: "compute", priority: 0.8 },
      { id: "normalization-node", type: "compute", priority: 0.6 },
      { id: "activation-node", type: "compute", priority: 0.5 }
    ];
    
    for (const point of defaultPoints) {
      this.pointRoutingTable.set([point.id](http://point.id), {
        id: [point.id](http://point.id),
        type: point.type,
        priority: point.priority,
        metadata: {
          origin_signature: "MrLiouWord",
          createdAt: new Date().toISOString()
        }
      });
      
      this.routingGraph.addNode([point.id](http://point.id), { type: point.type, priority: point.priority });
    }
    
    // 線路由 - 從節點到節點的連結
    const defaultLines = [
      { id: "attn-to-norm", from: "attention-node", to: "normalization-node", weight: 0.7 },
      { id: "norm-to-act", from: "normalization-node", to: "activation-node", weight: 0.8 },
      { id: "matmul-to-norm", from: "matmul-node", to: "normalization-node", weight: 0.6 }
    ];
    
    for (const line of defaultLines) {
      this.lineRoutingTable.set([line.id](http://line.id), {
        id: [line.id](http://line.id),
        fromId: line.from,
        toId: [line.to](http://line.to),
        weight: line.weight,
        metadata: {
          origin_signature: "MrLiouWord",
          createdAt: new Date().toISOString()
        }
      });
      
      this.routingGraph.addEdge(line.from, [line.to](http://line.to), { id: [line.id](http://line.id), weight: line.weight });
    }
    
    // 面路由 - 高階結構路由（複雜層次結構）
    const defaultSurfaces = [
      { 
        id: "attention-flow", 
        nodes: ["attention-node", "normalization-node", "activation-node"],
        density: 0.6
      }
    ];
    
    for (const surface of defaultSurfaces) {
      this.surfaceRoutingTable.set([surface.id](http://surface.id), {
        id: [surface.id](http://surface.id),
        nodeIds: surface.nodes,
        density: surface.density,
        metadata: {
          origin_signature: "MrLiouWord",
          createdAt: new Date().toISOString()
        }
      });
      
      // 將面結構路由添加到路由圖中（作為高階屬次結構）
      this.routingGraph.addSurface([surface.id](http://surface.id), surface.nodes, { density: surface.density });
    }
  }
  
  /**
   * 註冊新的點
```