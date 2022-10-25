from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, Dataset
import pandas
from sklearn import model_selection, preprocessing
import torch.nn.functional as F


# 0.load data
df = pandas.read_csv("./ml-latest-small/ratings.csv")

# 1.prepare dataset
class MovieData(Dataset):
    def __init__(self, users, movies, ratings) -> None:
        super().__init__()
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):

        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "movie": torch.tensor(movie, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float32),
        }


# 2.create model
class RecommandModel(torch.nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()

        self.user_embed = torch.nn.Embedding(n_users, 32)
        self.movie_embed = torch.nn.Embedding(n_movies, 32)
        self.linear1 = torch.nn.Linear(64, 128)
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, users, movies, rating=None):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        keep_going = torch.cat([user_embeds, movie_embeds], dim=1)
        keep_going = F.relu(self.linear1(keep_going))
        keep_going = self.linear2(keep_going)
        return keep_going


# 3.make data ready for training
ibl_user = preprocessing.LabelEncoder()
ibl_movie = preprocessing.LabelEncoder()

df.userId = ibl_user.fit_transform(df.userId.values)
df.movieId = ibl_movie.fit_transform(df.movieId.values)

df_train, df_valid = model_selection.train_test_split(
    df, test_size=0.1, random_state=1, stratify=df.rating.values
)

train_dataset = MovieData(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values,
)

valid_dataset = MovieData(
    users=df_valid.userId.values,
    movies=df_valid.movieId.values,
    ratings=df_valid.rating.values,
)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=4, shuffle=True)

# 4.using the model
model = RecommandModel(n_users=len(ibl_user.classes_), n_movies=len(ibl_movie.classes_))
optim = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)
loss_fun = torch.nn.MSELoss()

# 5.training loop
epochs = 1
total_loss = 0
plot_step, print_step = 5000, 1000
step_count = 0
all_loss_list = []


model.train(mode=True)
for epoch in range(epochs):

    for i, train_data in enumerate(train_loader):

        output = model(train_data["user"], train_data["movie"])
        # reshape
        rating = train_data["rating"].view(4, -1).to(torch.float32)

        loss = loss_fun(output, rating)
        optim.zero_grad()
        loss.backward()
        optim.step()

        step_count += len(train_data["user"])

        if (step_count % plot_step) == 0:
            avg_loss = total_loss / len(train_data["user"]) * plot_step
            print(
                f"epoch:{epoch+1}/{epochs} || loss:{loss} || step:{step_count}/{90_000*(epochs)}"
            )
            all_loss_list.append(avg_loss)
            total_loss = 0


plt.figure()
plt.plot(all_loss_list)
plt.show()

# 6.evaluate model
model_output_list = []
target_rating_list = []

model.eval()
with torch.no_grad():

    for i, batched_data in enumerate(valid_loader):

        model_output = model(batched_data["user"], batched_data["movie"])
        model_output_list.append(model_output.sum().item() / len(batched_data["user"]))
        target_rating = batched_data["rating"]

        target_rating_list.append(
            target_rating.sum().item() / len(batched_data["rating"])
        )

        print(f"model_output:{model_output} || target_output:{target_rating}")


# eval by another method
user_est_true = defaultdict(list)

with torch.no_grad():
    for i, batched_data in enumerate(valid_loader):
        users = batched_data["user"]
        movie = batched_data["movie"]
        ratings = batched_data["rating"]

        model_output = model(batched_data["user"], batched_data["movie"])

        for i in range(len(users)):
            user_id = users[i].item()
            movie_id = movie[i].item()
            pred_rating = model_output[i][0].item()
            true_rating = ratings[i].item()

            print(
                f"user:{user_id} || movie:{movie_id} ||pred:{pred_rating} || actual:{true_rating}"
            )
            user_est_true["user_id"].append((pred_rating, true_rating))
rms = mean_squared_error(target_rating_list, model_output_list, squared=False)
print(f"rms:{rms}")


#🐋rms:0.4954375100365766 ⭕