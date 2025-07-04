# System Architecture

Describe the overall architecture with user, controller, agents.

🌐 User Panel ──►  XControl API  ◄───────┐
     ▲                │                  │
     │ REST API       ▼                  │
[浏览器]        ┌─────────────┐        gRPC/HTTP
                │ PostgreSQL  │        ┌────────────────────┐
                └─────────────┘        │ Xray 节点 Agent #1 │
                                       │ Xray + stats/api    │
                                       └────────────────────┘
                                       ...
                                       ┌────────────────────┐
                                       │ DeepFlow Agent #N  │
                                       │ eBPF + WASM采流量    │
                                       └────────────────────┘



🧭 多租户 VLESS 管理系统架构图 （用户面 + 控制面 + 多节点）

                       🌐 用户面（User Panel）
┌─────────────────────────────────────────────────────────────┐
│ [User Browser]                                              │
│   ├─📄 查看配置导出（vless:// + QR 码）                     │
│   ├─📊 当前使用量（上/下行流量、图表）                     │
│   └─🌍 可用节点列表（选择订阅）                             │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTPS API 请求
               ▼

                       🧠 控制面（vless-admin Controller）
┌─────────────────────────────────────────────────────────────┐
│ REST API (Gin)                                              │
│   ├─ /api/users         → 用户注册/添加/流量查询           │
│   ├─ /api/subscription  → vless:// 订阅链接生成            │
│   ├─ /api/nodes         → 多节点信息展示                   │
│   └─ /api/stats         → 后台流量采集、流控策略           │
│                                                             │
│ PostgreSQL                                                  │
│   ├─ 用户表 (email + UUID)                                 │
│   ├─ 节点表                                                 │
│   └─ 流量表（每日/每小时）                                 │
│                                                             │
│ WebUI (Vue3) → embed.FS（Go 二进制嵌入）                    │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTP/gRPC 控制与拉取配置
               ▼

                     🛰️ 多节点（Xray-core + Agent）

┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Agent: Node #1     │   │ Agent: Node #2     │   │ Agent: Node #N     │
│ ┌────────────────┐ │   │ ┌────────────────┐ │   │ ┌────────────────┐ │
│ │ Xray-core      │ │   │ │ Xray-core      │ │   │ │ Xray-core      │ │
│ │ + stats + api  │ │   │ │ + stats + api  │ │   │ │ + stats + api  │ │
│ └────────────────┘ │   │ └────────────────┘ │   │ └────────────────┘ │
│  ⬆ 上报用户流量     │   │  ⬆ 上报用户流量     │   │  ⬆ 上报用户流量     │
│  ⬇ 拉取用户配置     │   │  ⬇ 拉取用户配置     │   │  ⬇ 拉取用户配置     │
└────────────────────┘   └────────────────────┘   └────────────────────┘


