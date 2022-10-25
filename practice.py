from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pandas
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


df = pandas.read_csv("./ml-latest-small/ratings.csv")


class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings) -> None:
        super().__init__()

        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        rating = self.ratings[index]

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "movie": torch.tensor(movie, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float32),
        }


class RecommandModel(torch.nn.Module):
    def __init__(self, n_users, n_movies) -> None:
        super().__init__()

        self.user_embed = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=32)
        self.movies_embed = torch.nn.Embedding(
            num_embeddings=n_movies, embedding_dim=32
        )
        self.leak_relu = torch.nn.LeakyReLU()
        self.linear1 = torch.nn.Linear(in_features=64, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=256)
        self.linear3 = torch.nn.Linear(in_features=256, out_features=1)

    def forward(self, user, movie, rating=None):
        user_embed = self.user_embed(user)
        movie_embed = self.movies_embed(movie)
        keep_going = torch.concat([user_embed, movie_embed], dim=1)
        keep_going = self.leak_relu(self.linear1(keep_going))
        keep_going = self.leak_relu(self.linear2(keep_going))
        keep_going = self.linear3(keep_going)
        return keep_going


ibl_user = preprocessing.LabelEncoder()
ibl_movie = preprocessing.LabelEncoder()

df.userId = ibl_user.fit_transform(df.userId.values)
df.movieId = ibl_movie.fit_transform(df.movieId.values)

df_train, df_valid = train_test_split(
    df, test_size=0.1, random_state=1, shuffle=True, stratify=df.rating.values
)

train_data = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values,
)

valid_data = MovieDataset(
    users=df_valid.userId.values,
    movies=df_valid.movieId.values,
    ratings=df_valid.rating.values,
)

train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=True)

model = RecommandModel(len(ibl_user.classes_), len(ibl_movie.classes_))
optim = torch.optim.SGD(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optim, gamma=0.05, milestones=[5, 10]
)
loss_fun = torch.nn.MSELoss()

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
        target_rating = train_data["rating"].view(4, -1).to(torch.float32)

        loss = loss_fun(output, target_rating)

        loss.backward()
        optim.step()

        optim.zero_grad()

        step_count += len(train_data["user"])

        if step_count % plot_step == 0:
            avg_loss = total_loss / (len(train_data["user"]) * plot_step)

            print(
                f"epoch:{epoch}/{epochs} || loss:{loss} || step:{step_count}/{epochs*90000}"
            )
            all_loss_list.append(avg_loss)
            total_loss = 0


model_output_list = []
target_rating_list = []

model.eval()
with torch.no_grad():

    for i, batched_data in enumerate(valid_loader):

        output = model(batched_data["user"], batched_data["movie"])
        model_output_list.append(output.sum().item() / len(batched_data["user"]))

        target_rating = batched_data["rating"]
        target_rating_list.append(
            target_rating.sum().item() / len(batched_data["rating"])
        )

        print(f"model_output:{output} || target_output:{target_rating}")

user_est_true = defaultdict(list)

with torch.no_grad():
    for i, batched_data in enumerate(valid_loader):
        users = batched_data["user"]
        movies = batched_data["movie"]
        ratings = batched_data["rating"]

        model_output = model(batched_data["user"], batched_data["movie"])

        for i in range(len(users)):
            user_id = users[i]
            movie_id = movies[i]
            pred_rating = model_output[i][0]
            target_rating = ratings[i]

            print(
                f"user: {user_id} || movie: {movie_id} || pred: {pred_rating} || target:{target_rating}"
            )
            user_est_true["userId"].append((pred_rating, target_rating))

rms = mean_squared_error(y_true=target_rating_list, y_pred=model_output_list)
print(f"🐋rms:{rms} ⭕")

# 🐋rms:0.24484403927990694 ⭕
#🐋rms:0.24438717761194173 ⭕
