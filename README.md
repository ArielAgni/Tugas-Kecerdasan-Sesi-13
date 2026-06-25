# Tugas-Kecerdasan-Sesi-13

***1.Apa yang dimaksud dengan State, Action, dan Reward dalam Reinforcement Learning?***

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

***2.Apa fungsi dari Learning Rate (α)?***

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

***3.Apa fungsi dari Discount Factor (γ)?***

Discount Factor (γ) menentukan seberapa penting reward di masa depan.

**gamma = 0.95**
```
reward + gamma * np.max(q_table[next_state])
```

Karena γ = 0.95, agent tidak hanya mempertimbangkan reward saat ini tetapi juga reward jangka panjang.


***4.Mengapa digunakan metode Exploration dan Exploitation?***

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

***5.Bagaimana perubahan nilai reward setelah training 2000 episode?***

* Sebelum 2000 Episode

Pada fase awal pelatihan, agent seringkali mendapatkan reward 0. Ini menunjukkan bahwa agent masih sering gagal mencapai tujuan atau jatuh ke dalam lubang es, karena masih dalam tahap eksplorasi dan belum menemukan jalur yang efektif.

* Setelah 2000 Episode

Menjelang akhir pelatihan,terlihat bahwa agent secara konsisten mendapatkan reward 1.0. Hal ini mengindikasikan bahwa agent sudah berhasil menemukan jalur optimal dan mampu menyelesaikan lingkungan FrozenLake hampir di setiap episode. Ini menunjukkan bahwa proses pembelajaran Q-Learning telah berhasil membuat agent menjadi sangat efektif.

## Modifikasi

***Menggunakan environment Taxi-v3 Menampilkan rata-rata reward setiap 100 episode Membandingkan hasil training 1000, 2000, dan 5000 episode***

**1000 Episode**

<img width="1062" height="588" alt="1000" src="https://github.com/user-attachments/assets/6cdc90dd-5439-4a69-825d-8a8398074337" />

Pada training sebanyak 1000 episode, agent mulai mempelajari lingkungan Taxi-v3 dan menemukan beberapa strategi yang baik. Namun, rata-rata reward masih belum stabil karena agent masih sering melakukan eksplorasi dan membuat kesalahan.

**2000 Episode**

<img width="1081" height="602" alt="2000" src="https://github.com/user-attachments/assets/295305c0-ae16-4d76-b103-9061b1dd9b36" />

Pada 2000 episode, performa agent meningkat dibandingkan 1000 episode. Agent lebih sering memilih aksi yang tepat sehingga rata-rata reward menjadi lebih tinggi dan proses penyelesaian tugas lebih efisien.

**5000 Episode**

<img width="455" height="252" alt="image" src="https://github.com/user-attachments/assets/9292bf36-0746-4811-ae53-14681234b0e3" />

Pada 5000 episode, agent telah memiliki pengalaman yang lebih banyak sehingga Q-Table menjadi lebih optimal. Rata-rata reward menjadi lebih stabil dan agent mampu menyelesaikan tugas dengan tingkat keberhasilan yang lebih baik dibandingkan training 1000 dan 2000 episode.
