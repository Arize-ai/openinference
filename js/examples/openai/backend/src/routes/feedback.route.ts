import express, { Router } from "express";
import { feedback } from "../controllers/feedback.controller";

const router: Router = express.Router();

router.route("/").post(feedback);

export default router;
