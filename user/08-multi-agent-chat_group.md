# User: 'I would like to go to Paris.'
# Agent - FrontDesk: 'Visit the Louvre Museum.'
# Agent - Concierge: 'The recommendation to visit the Louvre Museum is quite popular and can often feel touristy. To refine the recommendation and provide a more local and authentic experience, consider suggesting an engaging activity in a less crowded area of the city, such as exploring hidden art galleries in the Le Marais district, or participating in a local cooking class. This way, the traveler can experience Paris's culture in a more intimate and unique setting.'
# Agent - FrontDesk: 'Visit Montmartre for its artistic vibe and stunning views.'
# Agent - Concierge: 'The recommendation to visit Montmartre is a step in the right direction, but it can still be quite touristy, especially around the Sacré-Cœur Basilica. To refine this suggestion, consider guiding the traveler to explore the lesser-known streets and hidden cafés in Montmartre, or perhaps connecting them with a local artist for a workshop. This would provide a more authentic and personal experience of the area, allowing them to see beyond the famous landmarks.'
# Agent - FrontDesk: 'Visit the hidden gems of Montmartre, like the Wall of Love.'
# Agent - Concierge: 'The recommendation to visit hidden gems in Montmartre, such as the Wall of Love, is an improvement and moves toward a more authentic experience. However, to enhance this suggestion further, encourage the traveler to wander through the quieter side streets, interact with local artisans, or participate in a community event if available. This way, they can immerse themselves even more deeply in the local culture and atmosphere.'
# Agent - FrontDesk: 'Visit the Wall of Love in Montmartre for a unique cultural experience.'
# Agent - Concierge: 'The recommendation to visit the Wall of Love in Montmartre is a nice touch, but it still leans towards a tourist attraction. To improve this suggestion and create a more authentic experience, consider recommending a nearby local café or artisan shop where the traveler can relax and engage with the community, or suggest attending an event at a local venue that features traditional music or art. This would allow for a deeper connection to the local culture beyond just visiting a landmark.'
# Agent - FrontDesk: 'I can only provide one recommendation per response. Please feel free to ask for another suggestion!'
# Agent - Concierge: 'It sounds like the front desk is looking to keep things concise. However, for a truly enriching experience, it would be beneficial to encourage travelers to explore multiple suggestions or a thematic itinerary. This way, they can gain a broader understanding of the local culture, making their visit much more immersive and memorable.'
# IS COMPLETE: False

这个多 Agent 协作示例是一个**典型的目标驱动对话系统**，通过两个角色：

* `FrontDesk`：负责提出旅游推荐（建议）
* `Concierge`：对推荐进行审核，并提供反馈（判断是否地道/本地化/非游客向）

## 🎯 核心目标

**判断前台推荐的旅游项目是否达到了“地道非游客向”的标准**，一旦 `Concierge` 明确表示“approved”，对话终止。

---

## 🧠 多 Agent 协作机制详解

### 1. 角色分工明确：

| Agent       | 角色定位         | 行为限制                |
| ----------- | ------------ | ------------------- |
| `FrontDesk` | 简洁的旅行前台推荐人员  | 每次只给一个推荐；简短直接，不扯闲话  |
| `Concierge` | 酒店礼宾，有地道文化品味 | 判断推荐是否地道，未通过则建议如何改进 |

### 2. 控制策略设定：

* `selection_function`: 控制谁发言（轮流交替）
* `termination_function`: 控制是否终止（是否收到明确 approval）

---

## 🔁 实际对话过程分析

以下是对话节选摘要与评估：

| 回合 | FrontDesk 推荐             | Concierge 反馈                     | 是否认可？ |
| -- | ------------------------ | -------------------------------- | ----- |
| 1  | 卢浮宫                      | 太游客向，建议改为 Le Marais 等地道体验        | ❌     |
| 2  | 蒙马特                      | 方向对了，但太热门，建议深入小街或工作坊             | ❌     |
| 3  | Wall of Love             | 改进了，但仍是打卡地，建议结合本地互动体验            | ❌     |
| 4  | 仍是 Wall of Love，加上文化体验定位 | 仍偏打卡，建议再丰富推荐内容                   | ❌     |
| 5  | 提示 Concierge：只接受单一推荐     | Concierge：建议做 thematic itinerary | ❌     |
