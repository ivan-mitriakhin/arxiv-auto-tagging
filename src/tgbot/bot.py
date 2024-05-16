from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, ConversationHandler, filters
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from inference import make_prediction

data = {'title': '', 'abstract': ''}
TITLE, ABSTRACT, PREDICT = range(3)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}! It seems like you want me to tag your paper. Write the title of your paper:')

    return TITLE

async def get_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data["title"] = update.message.text

    await update.message.reply_text(f'Write the abstract of your paper: ')

    return ABSTRACT

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    data["abstract"] = update.message.text
    tags = make_prediction(data["title"], data["abstract"])
    tags = ', '.join(tags)

    await update.message.reply_text(f'The predicted tags are {tags}.')
    
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(f"Have a nice day. Bye!")

    return ConversationHandler.END

if __name__ == '__main__':
    load_dotenv()

    app = ApplicationBuilder().token(os.getenv("TOKEN")).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
        ],
        states={
            TITLE: [MessageHandler(filters.ALL, get_title)],
            ABSTRACT: [MessageHandler(filters.ALL, predict)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)

    app.run_polling()