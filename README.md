# Tugas-Kecerdasan-Sesi-13

1.Apa yang dimaksud dengan State, Action, dan Reward dalam Reinforcement Learning?

**State**

* adalah kondisi atau posisi agent dalam environment. Pada kode, jumlah state ditampilkan dengan:
```
print("Jumlah State :", env.observation_space.n)
```
**Action**

* adalah tindakan yang dapat dipilih agent. Pada kode:
```
print("Jumlah Action :", env.action_space.n)
```
**Reward**
* adalah nilai yang diterima agent setelah melakukan action. Reward diperoleh dari:
```
next_state, reward, terminated, truncated, info = env.step(action)
```

2.Apa fungsi dari Learning Rate (α)?

Learning Rate (α) mengatur seberapa besar pengalaman baru memengaruhi nilai Q yang sudah ada.

**alpha = 0.8**

```
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards.append(total_reward)

print("Training selesai!")
```
Karena α = 0,8, maka agent memberikan bobot yang cukup besar terhadap pengalaman baru yang diperoleh selama training.

3.Apa fungsi dari Discount Factor (γ)?

Discount Factor (γ) menentukan seberapa penting reward di masa depan.

**gamma = 0.95**
```
reward + gamma * np.max(q_table[next_state])
```

Karena γ = 0.95, agent tidak hanya mempertimbangkan reward saat ini tetapi juga reward jangka panjang.


4.Mengapa digunakan metode Exploration dan Exploitation?

Exploration digunakan agar agent mencoba action baru dan mengenal environment.

```
action = env.action_space.sample()
```
Exploitation digunakan agar agent memilih action terbaik berdasarkan Q-Table.

```
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
```
Keduanya diperlukan agar agent dapat menemukan sekaligus memanfaatkan jalur terbaik menuju goal.

5.Bagaimana perubahan nilai reward setelah training 2000 episode?

**episodes = 2000**
```
rewards.append(total_reward)
```

Pada awal training, reward sering bernilai 0 karena agent masih belajar. Setelah mendekati 2000 episode, Q-Table semakin baik sehingga agent lebih sering mencapai goal. Akibatnya reward menjadi lebih tinggi dan lebih stabil dibandingkan pada awal training.
