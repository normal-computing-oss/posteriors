# Extended Kalman filter for Premier League football

In this example, we'll use the extended Kalman filter implementation from `posteriors`
to infer the skills of Premier League football teams. We'll use a simple Elo-style
Bayesian model from [Duffield et al](https://arxiv.org/abs/2308.02414) to model the
outcome of matches.

## Data

We'll use the [football-data.co.uk](https://www.football-data.co.uk/englandm.php) and 
[`pandas`](https://pandas.pydata.org/) to load the Premier League results for the
2021/22 and 2022/2023 seasons.

??? quote "Code to download Premier League data"
    ```python
    import torch
    import matplotlib.pyplot as plt
    import pandas as pd
    import posteriors


    def download_data(start=21, end=23):
        urls = [
            f"https://www.football-data.co.uk/mmz4281/{y}{y+1}/E0.csv"
            for y in range(start, end)
        ]

        origin_date = pd.to_datetime(f"20{start}-08-01")
        data = pd.concat(pd.read_csv(url) for url in urls)
        data = data.dropna()
        data["Timestamp"] = pd.to_datetime(data["Date"], dayfirst=True)
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit="D")
        data["TimestampDays"] = (data["Timestamp"] - origin_date).dt.days.astype(int)

        players_arr = pd.unique(pd.concat([data["HomeTeam"], data["AwayTeam"]]))
        players_arr.sort()
        players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
        players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

        data["HomeTeamID"] = data["HomeTeam"].apply(lambda s: players_name_to_id_dict[s])
        data["AwayTeamID"] = data["AwayTeam"].apply(lambda s: players_name_to_id_dict[s])

        match_times = torch.tensor(data["TimestampDays"].to_numpy(), dtype=torch.float64)
        match_player_indices = torch.tensor(data[["HomeTeamID", "AwayTeamID"]].to_numpy())

        home_goals = torch.tensor(data["FTHG"].to_numpy())
        away_goals = torch.tensor(data["FTAG"].to_numpy())

        match_results = torch.where(
            home_goals > away_goals, 1, torch.where(home_goals < away_goals, 2, 0)
        )

        dataset = torch.utils.data.StackDataset(
            match_times=match_times,
            match_player_indices=match_player_indices,
            match_results=match_results,
        )

        return (
            dataset,
            players_id_to_name_dict,
            players_name_to_id_dict,
        )
    ```

We can load the dataset into a torch `DataLoader` as follows:

```python
(
    dataset,
    players_id_to_name_dict,
    players_name_to_id_dict,
) = download_data()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
```

## Define the model

We'll define a likelihood function that maps the skills of the teams to the probability
of a draw, home win or away win. Following [Duffield et al](https://arxiv.org/abs/2308.02414),
we'll use a simple sigmoidal model:

\begin{aligned}
p\left(y \mid x^h, x^a\right) 
=
\begin{cases}
    \textbf{sigmoid} \left(x^h - x^a + \epsilon\right) - \textbf{sigmoid}\left(x^h - x^a - \epsilon \right) & \text{if } y = \text{draw}, \\
    \textbf{sigmoid} \left(x^h - x^a - \epsilon \right) & \text{if } y = \mathrm{h},\\
    1-\textbf{sigmoid} \left(x^h - x^a + \epsilon \right) & \text{if } y = \mathrm{a}.
\end{cases}
\end{aligned}

Where $y$ is the match result, $x^h$ and $x^a$ are the (latent) skills of the home and
away teams. $\epsilon$ is a static parameter controlling the probability of a draw and 
$\textbf{sigmoid}$ is a [sigmoid function](https://pytorch.org/docs/stable/generated/torch.sigmoid.html)
that maps real values to the interval $[0, 1]$.


In code:
```python
epsilon = 1.0

def log_likelihood(params, batch):
    player_indices = batch["match_player_indices"]
    match_results = batch["match_results"]

    player_skills = params[player_indices]

    home_win_prob = torch.sigmoid(player_skills[:, 0] - player_skills[:, 1] - epsilon)
    away_win_prob = 1 - torch.sigmoid(
        player_skills[:, 0] - player_skills[:, 1] + epsilon
    )
    draw_prob = 1 - home_win_prob - away_win_prob
    result_probs = torch.vstack([draw_prob, home_win_prob, away_win_prob]).T
    log_liks = torch.log(result_probs[torch.arange(len(match_results)), match_results])
    return log_liks, result_probs
```
We've chosen $\epsilon = 1.0$ for this example, this gives a draw probability≈0.5 for
equally skilled teams which seems reasonable but ideally we'd like to learn this
parameter from data too.


## Extended Kalman time!
Now we'll run an extended Kalman filter to infer the skills of the teams sequentially
over the matches.

Because the matches are not equally spaced in time, we'll use a `transition_sd`
parameter that varies as we move through the matches. This means we can't use
`posteriors.ekf.diag_fisher.build` since this would globally configure the `transition_sd`,
luckily we can use `posteriors.ekf.diag_fisher.init` and `posteriors.ekf.diag_fisher.update`
directly instead.

```python
transition_sd_scale = 0.1
num_teams = len(players_id_to_name_dict)

init_means = torch.zeros((num_teams,))
init_sds = torch.ones((num_teams,))

state = posteriors.ekf.diag_fisher.init(init_means, init_sds)
all_means = init_means.unsqueeze(0)
all_sds = init_sds.unsqueeze(0)
previous_time = 0.0
for match in dataloader:
    match_time = match["match_times"]
    state, aux = posteriors.ekf.diag_fisher.update(
        state,
        match,
        log_likelihood,
        lr=1.0,
        per_sample=True,
        transition_sd=torch.sqrt(transition_sd_scale**2 * (match_time - previous_time)),
    )
    all_means = torch.vstack([all_means, state.params.unsqueeze(0)])
    all_sds = torch.vstack([all_sds, state.sd_diag.unsqueeze(0)])
    previous_time = match_time
```
Again we'd ideally like to learn the `transition_sd_scale` parameter from data, but for
this example we'll set it to 0.1 and see how it looks.

Here we've stored the means and standard deviations of the skills of the teams at each
time step, so we can plot them. This is doable as there's only 20 teams in the Premier
League.

## Plot the skills
We initialized the skills of the teams to zero, which is not realistic, so we'll use the
2021/22 season as a warm-up period and plot the skills of the teams in the 2022/23
season.

??? quote "Code to plot the skills"
    ```python
    last_season_start = (len(all_means) - 1) // 2
    times = dataset.datasets["match_times"][last_season_start:]
    means = all_means[last_season_start + 1 :]
    sds = all_sds[last_season_start + 1 :]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(num_teams):
        team_name = players_id_to_name_dict[i]
        if team_name in ("Arsenal", "Man City"):
            c = "skyblue" if team_name == "Man City" else "red"
            ax.plot(times, means[:, i], c=c, zorder=2, label=team_name)
            ax.fill_between(
                times,
                means[:, i] - sds[:, i],
                means[:, i] + sds[:, i],
                color=c,
                alpha=0.3,
                zorder=1,
            )
        else:
            ax.plot(times, means[:, i], c="grey", alpha=0.2, zorder=0, linewidth=0.75)

    ax.set_xticklabels([])
    ax.set_xlabel("2022/23 season")
    ax.set_ylabel("Skill")
    ax.legend()
    fig.tight_layout()
    ```

![EKF Premier League](https://storage.googleapis.com/posteriors/ekf_premier_league.png)

Here we can see the inferred skills (and one standard deviation) for Arsenal and
Manchester City with the skills of the other teams in grey. We can see that the extended
Kalman filter has inferred that Arsenal were the strongest team through most of the
season before deteriorating towards the end of the season where Manchester City came
through as champions. Note the vacant period mid-season where the 2022 world cup took
place.

Observe the standard deviations increasing between matches, this is natural due to the
increasing uncertainty as we move further from the last match. But as we observe more
matches, the uncertainty decreases.

!!! note
    The raw code for this example can be found in the repo at [examples/ekf_premier_league.py](https://github.com/normal-computing-oss/posteriors/blob/main/examples/ekf_premier_league.py).



